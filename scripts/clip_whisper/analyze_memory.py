#!/usr/bin/env python3
"""
Script to analyze GPU memory usage of ClipWhisperModel components.
"""

import os
import sys
import argparse
import torch
import logging
import json
from pathlib import Path
import gc
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.clip_whisper.models import ClipWhisperModel
from src.clip_whisper.models.modality_connector import ModalityConnector
from transformers import WhisperModel, WhisperProcessor, CLIPModel, CLIPProcessor, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def setup_logging(log_level="info"):
    """Setup logging with specified level"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "total": 0}
    
    # Force a CUDA synchronization to ensure memory is properly reported
    torch.cuda.synchronize()
    
    return {
        "allocated": torch.cuda.memory_allocated() / (1024 * 1024),
        "reserved": torch.cuda.memory_reserved() / (1024 * 1024),
        "total": torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    }


def clear_gpu_memory():
    """Clear GPU cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    else:
        gc.collect()


def measure_memory_usage(callback):
    """Measure memory usage before and after executing callback"""
    # Clear memory before measurement
    clear_gpu_memory()
    
    # Measure initial memory
    initial = get_gpu_memory()
    
    # Execute callback
    result = callback()
    
    # Measure final memory
    final = get_gpu_memory()
    
    # Calculate difference
    diff = {
        "allocated": final["allocated"] - initial["allocated"],
        "reserved": final["reserved"] - initial["reserved"],
    }
    
    return diff, result


def load_whisper_standalone(model_name, freeze=True):
    """Load Whisper model without requiring ClipWhisperModel instance"""
    try:
        logging.info(f"Loading Whisper model: {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Clear GPU cache before loading
        clear_gpu_memory()
        
        # Load model and move to GPU
        model = WhisperModel.from_pretrained(model_name).to(device)
        processor = WhisperProcessor.from_pretrained(model_name)
        
        if freeze:
            logging.info("Freezing Whisper model")
            for param in model.parameters():
                param.requires_grad = False
        
        # Force GPU sync to ensure memory is allocated
        if device == "cuda":
            torch.cuda.synchronize()
            
        return model, processor
    except Exception as e:
        logging.error(f"Error loading Whisper model: {e}")
        raise


def load_clip_standalone(model_name, freeze=True):
    """Load CLIP model without requiring ClipWhisperModel instance"""
    try:
        logging.info(f"Loading CLIP model: {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Clear GPU cache before loading
        clear_gpu_memory()
        
        # Load model and move to GPU
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        
        if freeze:
            logging.info("Freezing CLIP model")
            for param in model.parameters():
                param.requires_grad = False
                
        # Force GPU sync to ensure memory is allocated
        if device == "cuda":
            torch.cuda.synchronize()
            
        return model, processor
    except Exception as e:
        logging.error(f"Error loading CLIP model: {e}")
        raise


def load_llm_standalone(model_path, use_lora=True, lora_r=16, lora_alpha=32, 
                       lora_dropout=0.05, freeze=False, use_4bit=False):
    """Load LLM without requiring ClipWhisperModel instance"""
    try:
        logging.info(f"Loading LLM model: {model_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configure 4-bit quantization if requested
        if use_4bit:
            try:
                import bitsandbytes
                logging.info("Using 4-bit quantization for LLM")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                llm = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    quantization_config=quantization_config,
                )
                
                logging.info("Successfully loaded LLM with 4-bit quantization")
            except ImportError:
                logging.warning("BitsAndBytes not available for 4-bit quantization, falling back to standard loading")
                llm = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
            except Exception as e:
                logging.error(f"Error loading LLM with 4-bit quantization: {e}")
                logging.warning("Falling back to standard loading")
                llm = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        else:
            # Standard loading without quantization
            llm = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Apply LoRA if requested
        if use_lora:
            from peft import LoraConfig, get_peft_model
            logging.info(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            
            # Configure target modules based on model type
            if 'llama' in model_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            else:
                # Default target modules for other model types
                target_modules = ["query", "key", "value", "dense"]
                
            # Create LoRA config
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            
            # Apply LoRA to model
            llm = get_peft_model(llm, peft_config)
            logging.info("LoRA applied successfully")
            
        # Freeze LLM weights if requested
        if freeze:
            logging.info("Freezing LLM weights")
            for param in llm.parameters():
                param.requires_grad = False
                
            # If using LoRA, ensure LoRA weights are trainable
            if use_lora:
                for n, p in llm.named_parameters():
                    if 'lora' in n:
                        p.requires_grad = True
                        
        return llm, tokenizer
    
    except Exception as e:
        logging.error(f"Error loading LLM: {e}")
        raise


def get_llm_dim(llm):
    """Get the input dimension of the LLM"""
    if hasattr(llm, "get_input_embeddings"):
        embedding = llm.get_input_embeddings()
        if hasattr(embedding, "weight"):
            return embedding.weight.shape[1]
    
    # Default dimension for common LLMs
    return 4096


def convert_to_native_types(obj):
    """Convert NumPy values to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_types(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


def measure_projection_memory_usage(whisper=None, clip=None, llm=None):
    """Measure memory usage of projection layers"""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0}
    
    try:
        # Skip if any required model is missing
        if llm is None or (whisper is None and clip is None):
            return {"allocated": 0, "reserved": 0}
        
        device = "cuda"
        clear_gpu_memory()
        
        # Get dimensions
        llm_dim = get_llm_dim(llm)
        logging.info(f"LLM input dimension: {llm_dim}")
        
        # Create input dimensions
        if whisper is not None:
            whisper_dim = whisper.encoder.config.d_model
            logging.info(f"Whisper encoder dimension: {whisper_dim}")
        else:
            whisper_dim = None
            
        if clip is not None:
            clip_dim = clip.vision_model.config.hidden_size
            logging.info(f"CLIP vision dimension: {clip_dim}")
        else:
            clip_dim = None
        
        initial = get_gpu_memory()
        projections = {}
        
        # Create audio projection if whisper is available
        if whisper_dim is not None:
            audio_connector = ModalityConnector(
                input_dim=whisper_dim,
                output_dim=llm_dim
            ).to(device)
            projections["audio"] = audio_connector
            
        # Create video projection if clip is available
        if clip_dim is not None:
            video_connector = ModalityConnector(
                input_dim=clip_dim,
                output_dim=llm_dim
            ).to(device)
            projections["video"] = video_connector
            
        final = get_gpu_memory()
        
        # Calculate memory usage
        diff = {
            "allocated": final["allocated"] - initial["allocated"],
            "reserved": final["reserved"] - initial["reserved"],
        }
        
        return diff
        
    except Exception as e:
        logging.error(f"Error measuring projection memory: {e}")
        return {"allocated": 0, "reserved": 0}


def load_components_separately(args):
    """Load model components one by one to measure memory usage"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    memory_stats = {"components": {}, "total": {"used": 0, "available": 0}, "modality": args.modality}
    
    if torch.cuda.is_available():
        memory_stats["total"]["available"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    
    print("\n" + "="*80)
    print(f"ANALYZING GPU MEMORY USAGE FOR CLIP-WHISPER MODEL ({args.modality.upper()} MODE)")
    print("="*80)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE - Cannot measure GPU memory usage")
        return memory_stats
    
    # Measure baseline memory usage (PyTorch itself)
    baseline = get_gpu_memory()
    memory_stats["baseline"] = baseline
    
    print(f"\nBaseline GPU memory (PyTorch overhead): {baseline['allocated']:.2f} MB")
    
    # Measure Whisper memory usage if audio modality is enabled
    whisper_memory = {"allocated": 0, "reserved": 0}
    whisper_model = None
    whisper_processor = None
    
    if args.modality in ["audio", "multimodal"] and args.whisper_model:
        print("\nLoading Whisper model...")
        whisper_memory, whisper_result = measure_memory_usage(
            lambda: load_whisper_standalone(args.whisper_model, freeze=True)
        )
        memory_stats["components"]["whisper"] = whisper_memory
        print(f"Whisper model memory: +{whisper_memory['allocated']:.2f} MB")
        
        whisper_model, whisper_processor = whisper_result
    
    # Measure CLIP memory usage if video modality is enabled
    clip_memory = {"allocated": 0, "reserved": 0}
    clip_model = None
    clip_processor = None
    
    if args.modality in ["video", "multimodal"] and args.clip_model:
        print("\nLoading CLIP model...")
        clip_memory, clip_result = measure_memory_usage(
            lambda: load_clip_standalone(args.clip_model, freeze=True)
        )
        memory_stats["components"]["clip"] = clip_memory
        print(f"CLIP model memory: +{clip_memory['allocated']:.2f} MB")
        
        clip_model, clip_processor = clip_result
    
    # Measure LLM memory usage with different settings
    # First with standard loading
    print("\nLoading LLM (standard mode)...")
    llm_memory_standard, llm_result = measure_memory_usage(
        lambda: load_llm_standalone(
            args.llm_path, use_lora=args.use_lora, 
            lora_r=16, lora_alpha=32, lora_dropout=0.05, 
            freeze=False, use_4bit=False
        )
    )
    memory_stats["components"]["llm_standard"] = llm_memory_standard
    print(f"LLM memory (standard): +{llm_memory_standard['allocated']:.2f} MB")
    
    llm_model, llm_tokenizer = llm_result
    
    # Measure projection memory usage
    print("\nMeasuring projection layers memory...")
    projection_memory = measure_projection_memory_usage(
        whisper=whisper_model, 
        clip=clip_model, 
        llm=llm_model
    )
    memory_stats["components"]["projections"] = projection_memory
    print(f"Projection layers memory: +{projection_memory['allocated']:.2f} MB")
    
    # Clear models to free memory
    if whisper_model is not None:
        del whisper_model, whisper_processor
    if clip_model is not None:
        del clip_model, clip_processor
    del llm_model, llm_tokenizer
    clear_gpu_memory()
    
    # If 4-bit is available, measure with 4-bit quantization
    if args.check_4bit:
        try:
            import bitsandbytes
            print("\nLoading LLM (4-bit mode)...")
            llm_memory_4bit, _ = measure_memory_usage(
                lambda: load_llm_standalone(
                    args.llm_path, use_lora=args.use_lora, 
                    lora_r=16, lora_alpha=32, lora_dropout=0.05, 
                    freeze=False, use_4bit=True
                )
            )
            memory_stats["components"]["llm_4bit"] = llm_memory_4bit
            print(f"LLM memory (4-bit): +{llm_memory_4bit['allocated']:.2f} MB")
            print(f"Memory savings with 4-bit: {llm_memory_standard['allocated'] - llm_memory_4bit['allocated']:.2f} MB " +
                  f"({(1 - llm_memory_4bit['allocated'] / llm_memory_standard['allocated']) * 100:.1f}%)")
        except ImportError:
            print("bitsandbytes not installed, skipping 4-bit measurement")
    
    clear_gpu_memory()
    
    # Calculate total memory usage for standard case
    total_memory_standard = (
        baseline["allocated"] + 
        whisper_memory["allocated"] + 
        clip_memory["allocated"] + 
        llm_memory_standard["allocated"] + 
        projection_memory["allocated"]
    )
    memory_stats["total"]["used"] = total_memory_standard
    
    # Print memory usage summary
    print("\n" + "="*80)
    print("MEMORY USAGE SUMMARY")
    print("="*80)
    
    # Calculate percentages (avoid division by zero)
    if total_memory_standard > 0:
        baseline_pct = (baseline["allocated"] / total_memory_standard) * 100 if baseline["allocated"] > 0 else 0
        whisper_pct = (whisper_memory["allocated"] / total_memory_standard) * 100 if whisper_memory["allocated"] > 0 else 0
        clip_pct = (clip_memory["allocated"] / total_memory_standard) * 100 if clip_memory["allocated"] > 0 else 0
        llm_pct = (llm_memory_standard["allocated"] / total_memory_standard) * 100 if llm_memory_standard["allocated"] > 0 else 0
        proj_pct = (projection_memory["allocated"] / total_memory_standard) * 100 if projection_memory["allocated"] > 0 else 0
    else:
        baseline_pct = whisper_pct = clip_pct = llm_pct = proj_pct = 0
    
    print(f"Total GPU memory: {memory_stats['total']['available']:.2f} MB")
    print(f"Total model memory (standard): {total_memory_standard:.2f} MB")
    print(f"  - PyTorch overhead: {baseline['allocated']:.2f} MB ({baseline_pct:.1f}%)")
    
    if args.modality in ["audio", "multimodal"]:
        print(f"  - Whisper model: {whisper_memory['allocated']:.2f} MB ({whisper_pct:.1f}%)")
        
    if args.modality in ["video", "multimodal"]:
        print(f"  - CLIP model: {clip_memory['allocated']:.2f} MB ({clip_pct:.1f}%)")
        
    print(f"  - LLM (standard): {llm_memory_standard['allocated']:.2f} MB ({llm_pct:.1f}%)")
    print(f"  - Projection layers: {projection_memory['allocated']:.2f} MB ({proj_pct:.1f}%)")
    
    if args.check_4bit and "llm_4bit" in memory_stats["components"]:
        total_memory_4bit = (
            baseline["allocated"] + 
            whisper_memory["allocated"] + 
            clip_memory["allocated"] + 
            llm_memory_4bit["allocated"] + 
            projection_memory["allocated"]
        )
        memory_savings_pct = (1 - total_memory_4bit / total_memory_standard) * 100 if total_memory_standard > 0 else 0
        print(f"Total model memory (4-bit): {total_memory_4bit:.2f} MB")
        print(f"Memory savings with 4-bit: {total_memory_standard - total_memory_4bit:.2f} MB ({memory_savings_pct:.1f}%)")
    
    # Generate pie chart
    if args.output_chart:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Prepare data for the chart based on modality
        labels = ['PyTorch overhead']
        sizes = [baseline['allocated']]
        colors = ['gray']
        
        if args.modality in ["audio", "multimodal"]:
            labels.append('Whisper')
            sizes.append(whisper_memory['allocated'])
            colors.append('orange')
            
        if args.modality in ["video", "multimodal"]:
            labels.append('CLIP')
            sizes.append(clip_memory['allocated'])
            colors.append('green')
            
        labels.extend(['LLM', 'Projections'])
        sizes.extend([llm_memory_standard['allocated'], projection_memory['allocated']])
        colors.extend(['red', 'purple'])
        
        plt.figure(figsize=(12, 8))
        # Create a pie chart with a slight explosion for the largest component
        explode = [0] * len(sizes)
        if any(size > 0 for size in sizes):
            max_idx = sizes.index(max(sizes))
            explode[max_idx] = 0.1
        
        plt.pie(
            sizes, 
            labels=labels, 
            autopct='%1.1f%%', 
            startangle=90,
            explode=explode,
            colors=colors,
            shadow=True
        )
        plt.axis('equal')
        plt.title(f'GPU Memory Usage Breakdown ({args.modality.capitalize()} Mode)', fontsize=16)
        
        chart_path = os.path.join(args.output_dir, 'memory_usage_chart.png')
        plt.savefig(chart_path)
        print(f"\nSaved memory usage chart to: {chart_path}")
        
        # If 4-bit data is available, create a comparison chart
        if args.check_4bit and "llm_4bit" in memory_stats["components"]:
            standard_sizes = sizes.copy()
            
            # Replace the LLM size with 4-bit size
            llm_index = labels.index('LLM')
            fourbit_sizes = standard_sizes.copy()
            fourbit_sizes[llm_index] = llm_memory_4bit['allocated']
            
            # Create bar chart for comparison
            plt.figure(figsize=(14, 8))
            
            x = np.arange(len(labels))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(14, 8))
            rects1 = ax.bar(x - width/2, standard_sizes, width, label='Standard (FP32)')
            rects2 = ax.bar(x + width/2, fourbit_sizes, width, label='4-bit Quantization')
            
            ax.set_ylabel('Memory Usage (MB)', fontsize=14)
            ax.set_title(f'GPU Memory Usage by Component ({args.modality.capitalize()} Mode)', fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=12)
            ax.legend(fontsize=12)
            
            # Add memory values on top of bars
            def add_labels(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.1f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=10)
            
            add_labels(rects1)
            add_labels(rects2)
            
            # Add total memory usage
            ax.text(0.02, 0.95, f'Total (Standard): {total_memory_standard:.1f} MB', 
                    transform=ax.transAxes, fontsize=12, verticalalignment='top')
            ax.text(0.02, 0.90, f'Total (4-bit): {total_memory_4bit:.1f} MB', 
                    transform=ax.transAxes, fontsize=12, verticalalignment='top')
            ax.text(0.02, 0.85, f'Memory savings: {memory_savings_pct:.1f}%', 
                    transform=ax.transAxes, fontsize=12, verticalalignment='top')
            
            fig.tight_layout()
            
            comparison_chart_path = os.path.join(args.output_dir, 'memory_comparison_chart.png')
            plt.savefig(comparison_chart_path)
            print(f"Saved memory comparison chart to: {comparison_chart_path}")
    
    # Save memory stats to JSON
    if args.output_json:
        json_path = os.path.join(args.output_dir, 'memory_stats.json')
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Convert memory_stats to native Python types
        memory_stats_native = convert_to_native_types(memory_stats)
        
        with open(json_path, 'w') as f:
            json.dump(memory_stats_native, f, indent=2)
        print(f"Saved memory statistics to: {json_path}")
    
    print("\n" + "="*80)
    return memory_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze GPU memory usage of ClipWhisperModel components")
    parser.add_argument("--whisper_model", type=str, default="openai/whisper-medium",
                        help="Name or path of Whisper model")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32",
                        help="Name or path of CLIP model")
    parser.add_argument("--llm_path", type=str, default="checkpoints/Llama-3.2-1B",
                        help="Path or name of the LLM model")
    parser.add_argument("--output_dir", type=str, default="outputs/memory_analysis",
                        help="Directory to save outputs")
    parser.add_argument("--output_json", action="store_true", default=True,
                        help="Save memory statistics to JSON")
    parser.add_argument("--output_chart", action="store_true", default=True,
                        help="Generate a pie chart of memory usage")
    parser.add_argument("--check_4bit", action="store_true", default=True,
                        help="Also check memory usage with 4-bit quantization")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA when loading LLM")
    parser.add_argument("--modality", type=str, default="multimodal", choices=["audio", "video", "multimodal"],
                        help="Which modality to analyze (audio, video, or multimodal)")
    parser.add_argument("--log_level", type=str, default="info",
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging level")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    load_components_separately(args)


if __name__ == "__main__":
    main() 