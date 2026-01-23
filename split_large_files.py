import os
from pathlib import Path

# Max size in bytes (90MB to be safe for GitHub's 100MB limit)
MAX_SIZE_BYTES = 90 * 1024 * 1024

def split_csv_stream(file_path):
    """Split a CSV file into smaller parts using stream processing."""
    file_path = Path(file_path)
    file_size = file_path.stat().st_size
    
    if file_size <= MAX_SIZE_BYTES:
        print(f"Skipping {file_path.name} ({file_size / 1024 / 1024:.2f} MB) - Small enough")
        return

    print(f"\nSplitting {file_path.name} ({file_size / 1024 / 1024:.2f} MB)...")
    
    header = None
    part_num = 1
    current_out_file = None
    current_size = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8', newline='') as f_in:
            header = f_in.readline()
            
            # Create first part
            output_name = f"{file_path.stem}_part{part_num}{file_path.suffix}"
            output_path = file_path.parent / output_name
            current_out_file = open(output_path, 'w', encoding='utf-8', newline='')
            current_out_file.write(header)
            current_size = len(header.encode('utf-8'))
            
            print(f"  Writing {output_name}...", end='\r')
            
            for line in f_in:
                line_len = len(line.encode('utf-8'))
                
                # Check if adding this line exceeds limit
                if current_size + line_len > MAX_SIZE_BYTES:
                    current_out_file.close()
                    print(f"  Created {output_name} ({current_size/1024/1024:.2f} MB)")
                    
                    # Start new part
                    part_num += 1
                    output_name = f"{file_path.stem}_part{part_num}{file_path.suffix}"
                    output_path = file_path.parent / output_name
                    current_out_file = open(output_path, 'w', encoding='utf-8', newline='')
                    current_out_file.write(header)
                    current_size = len(header.encode('utf-8'))
                    print(f"  Writing {output_name}...", end='\r')
                
                current_out_file.write(line)
                current_size += line_len
            
            # Close last file
            if current_out_file:
                current_out_file.close()
                print(f"  Created {output_name} ({current_size/1024/1024:.2f} MB)")
                
        print(f"Done! Split into {part_num} parts.")
        
    except Exception as e:
        print(f"\nError splitting {file_path}: {e}")
        if current_out_file:
            current_out_file.close()

def main():
    # Directories to check
    dirs_to_check = [
        'data',
        'data/intel_by_year'
    ]
    
    for dir_name in dirs_to_check:
        if not os.path.exists(dir_name):
            continue
            
        print(f"\nScanning {dir_name}...")
        for filename in os.listdir(dir_name):
            if filename.endswith('.csv') and 'part' not in filename:
                file_path = os.path.join(dir_name, filename)
                split_csv_stream(file_path)

if __name__ == "__main__":
    main()
