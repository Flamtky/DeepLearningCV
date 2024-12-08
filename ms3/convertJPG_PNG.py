import os
from PIL import Image
from multiprocessing import Pool, cpu_count

def convert_image(jpg_path):
    try:
        # Create PNG path by replacing extension
        png_path = os.path.splitext(jpg_path)[0] + '.png'
        
        # If PNG exists, verify it and delete JPG
        if os.path.exists(png_path):
            try:
                with Image.open(png_path) as img:
                    # If we can open it, it's valid - delete JPG
                    os.remove(jpg_path)
                    print(f"PNG exists, deleted: {jpg_path}")
                return
            except:
                print(f"Existing PNG invalid for {png_path}, will reconvert")
                os.remove(png_path)
        
        # Open and convert image
        with Image.open(jpg_path) as img:
            img.save(png_path, 'PNG')
            
        # Verify PNG exists and is valid
        if os.path.exists(png_path):
            try:
                with Image.open(png_path) as img:
                    # If we can open it, it's valid
                    # Delete the original JPG
                    os.remove(jpg_path)
                    print(f"Converted and deleted: {jpg_path} -> {png_path}")
            except Exception as e:
                print(f"PNG verification failed for {png_path}")
                if os.path.exists(png_path):
                    os.remove(png_path)  # Remove invalid PNG
    except Exception as e:
        print(f"Error converting {jpg_path}: {str(e)}")

def get_jpg_files(directory):
    jpg_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                jpg_files.append(os.path.join(root, file))
    return jpg_files

def delete_all_jpg(directory):
    jpg_files = get_jpg_files(directory)
    for jpg_file in jpg_files:
        try:
            os.remove(jpg_file)
            print(f"Deleted: {jpg_file}")
        except Exception as e:
            print(f"Error deleting {jpg_file}: {str(e)}")
    print(f"Deleted {len(jpg_files)} JPG files")

if __name__ == '__main__':
    # Directory containing JPG files
    input_dir = '../train/faces'  # Adjust this path as needed
    
    # Get list of all JPG files
    jpg_files = get_jpg_files(input_dir)

    # Create a pool with number of CPU cores
    with Pool(processes=cpu_count()) as pool:
        # Convert images in parallel
        pool.map(convert_image, jpg_files)
    
    print("Conversion completed!")
    #delete_all_jpg(input_dir)
