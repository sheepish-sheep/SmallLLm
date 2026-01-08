"""
Upload checkpoint to Modal Volume.
Run this from your local Windows machine.
"""
import modal

# Create/get the volume
volume = modal.Volume.from_name("chuni-checkpoints", create_if_missing=True)

def upload_checkpoint():
    local_path = "log/model_04999.pt"
    remote_path = "model_04999.pt"
    
    print(f"Uploading {local_path} to Modal Volume...")
    
    with volume.batch_upload() as upload:
        upload.put_file(local_path, remote_path)
    
    print("Upload complete!")

def list_files():
    """Check what's in the volume."""
    for item in volume.listdir("/"):
        print(f"  {item.path} ({item.size / 1e9:.2f} GB)")

if __name__ == "__main__":
    upload_checkpoint()
    print("\nFiles in volume:")
    list_files()
