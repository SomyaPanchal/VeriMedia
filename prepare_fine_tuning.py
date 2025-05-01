#!/usr/bin/env python3
"""
Prepare fine-tuning data for VeriMedia toxicity classification.
This script processes labeled content and prepares it for OpenAI fine-tuning.
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def create_training_example(content, toxicity_level, content_type):
    """Create a formatted training example for OpenAI fine-tuning."""
    # Generate appropriate system message based on content type
    if content_type == "text":
        system_msg = "You are an expert analyzing text content for ethical standards. Classify the toxicity level."
        user_msg = f"Analyze the following text content for harmful language, misinformation, and problematic content. Content: {content}"
    elif content_type == "audio":
        system_msg = "You are an expert analyzing audio transcripts for ethical standards. Classify the toxicity level."
        user_msg = f"Analyze the following transcribed audio content for harmful language, misinformation, and problematic content. Transcribed content: {content}"
    elif content_type == "video":
        system_msg = "You are an expert analyzing video transcripts for ethical standards. Classify the toxicity level."
        user_msg = f"Analyze the following transcribed video content for harmful language, misinformation, and problematic content. Transcribed content: {content}"
    else:
        raise ValueError(f"Invalid content type: {content_type}")
    
    # Format the assistant's response with just the toxicity level
    assistant_msg = f"Toxicity level: {toxicity_level}"
    
    # Create the example in the required format
    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
    }

def process_csv_data(csv_path, content_column, label_column, type_column=None):
    """Process data from a CSV file containing content and labels."""
    df = pd.read_csv(csv_path)
    examples = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV data"):
        content = row[content_column]
        toxicity = row[label_column]
        
        # If type column is provided, use it; otherwise, default to "text"
        content_type = row.get(type_column, "text") if type_column else "text"
        
        # Skip rows with missing content
        if pd.isna(content) or content.strip() == "":
            continue
            
        example = create_training_example(content, toxicity, content_type)
        examples.append(example)
    
    return examples

def process_directory(dir_path):
    """Process files from a directory structure where folders are named by label."""
    examples = []
    dir_path = Path(dir_path)
    
    # Expect a structure like:
    # data/
    #   text/
    #     none/
    #     mild/
    #     high/
    #     max/
    #   audio/
    #     none/
    #     ...
    
    for content_type in ["text", "audio", "video"]:
        content_dir = dir_path / content_type
        if not content_dir.exists():
            continue
            
        for toxicity_level in ["None", "Mild", "High", "Max"]:
            level_dir = content_dir / toxicity_level.lower()
            if not level_dir.exists():
                continue
                
            for file_path in tqdm(list(level_dir.glob("*")), desc=f"Processing {content_type}/{toxicity_level}"):
                # Skip non-text files
                if not file_path.is_file():
                    continue
                    
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        
                    if content:
                        example = create_training_example(content, toxicity_level, content_type)
                        examples.append(example)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    return examples

def write_jsonl(examples, output_path):
    """Write examples to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            
    print(f"Wrote {len(examples)} examples to {output_path}")

def split_train_validation(examples, validation_ratio=0.1):
    """Split examples into training and validation sets."""
    import random
    random.shuffle(examples)
    
    split_idx = int(len(examples) * (1 - validation_ratio))
    train_examples = examples[:split_idx]
    valid_examples = examples[split_idx:]
    
    return train_examples, valid_examples

def main():
    parser = argparse.ArgumentParser(description="Prepare data for fine-tuning the toxicity classifier")
    parser.add_argument("--input", required=True, help="Input CSV file or directory with labeled content")
    parser.add_argument("--output", default="data/fine_tuning", help="Output directory for fine-tuning data")
    parser.add_argument("--input-type", choices=["csv", "directory"], required=True, help="Type of input")
    parser.add_argument("--content-col", default="content", help="Column name for content (CSV only)")
    parser.add_argument("--label-col", default="toxicity", help="Column name for toxicity label (CSV only)")
    parser.add_argument("--type-col", default=None, help="Column name for content type (CSV only)")
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Ratio of data to use for validation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process input data
    if args.input_type == "csv":
        examples = process_csv_data(args.input, args.content_col, args.label_col, args.type_col)
    else:
        examples = process_directory(args.input)
    
    print(f"Processed {len(examples)} examples")
    
    # Split into training and validation sets
    train_examples, valid_examples = split_train_validation(examples, args.validation_ratio)
    
    # Write output files
    write_jsonl(train_examples, os.path.join(args.output, "training.jsonl"))
    write_jsonl(valid_examples, os.path.join(args.output, "validation.jsonl"))
    
if __name__ == "__main__":
    main() 