"""
Hugging Face utilities for model uploads and management.
Handles model uploading, versioning, and metadata management.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, upload_file
    from transformers import AutoConfig, AutoModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("Hugging Face Hub not available. Install with: pip install huggingface_hub transformers")

class HuggingFaceManager:
    """Manages Hugging Face model uploads and repository operations."""
    
    def __init__(self, token: Optional[str] = None, repo_id: Optional[str] = None):
        """
        Initialize Hugging Face manager.
        
        Args:
            token: Hugging Face API token
            repo_id: Repository ID (username/repo_name)
        """
        self.token = token or os.getenv("HUGGINGFACE_TOKEN")
        self.repo_id = repo_id or os.getenv("HUGGINGFACE_REPO_ID")
        self.model_name = os.getenv("HUGGINGFACE_MODEL_NAME", "vision-transformer-mnist")
        
        if not self.token:
            raise ValueError("Hugging Face token not provided. Set HUGGINGFACE_TOKEN environment variable.")
        
        if not self.repo_id:
            raise ValueError("Repository ID not provided. Set HUGGINGFACE_REPO_ID environment variable.")
        
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face Hub not available. Install with: pip install huggingface_hub transformers")
        
        self.api = HfApi(token=self.token)
        self.logger = logging.getLogger(__name__)
    
    def create_repository(self, private: bool = False) -> bool:
        """
        Create a new repository on Hugging Face.
        
        Args:
            private: Whether the repository should be private
            
        Returns:
            True if successful, False otherwise
        """
        try:
            create_repo(
                repo_id=self.repo_id,
                token=self.token,
                private=private,
                exist_ok=True
            )
            self.logger.info(f"✅ Repository {self.repo_id} created/verified successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to create repository: {e}")
            return False
    
    def upload_model(
        self, 
        model_path: str, 
        model_config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Upload a trained model to Hugging Face.
        
        Args:
            model_path: Path to the model file
            model_config: Model configuration dictionary
            metadata: Additional metadata for the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"❌ Model file not found: {model_path}")
                return False
            
            # Create repository if it doesn't exist
            self.create_repository()
            
            # Prepare metadata
            model_metadata = {
                "model_name": self.model_name,
                "task": "image-classification",
                "dataset": "mnist",
                "architecture": "vision-transformer",
                "framework": "pytorch",
                "license": "mit",
                "tags": ["vision-transformer", "mnist", "image-classification"],
                **(metadata or {})
            }
            
            # Upload model file
            self.api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=f"{self.model_name}.pth",
                repo_id=self.repo_id,
                commit_message=f"Upload {self.model_name} model"
            )
            
            # Upload configuration
            config_path = model_path.replace('.pth', '_config.json')
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            
            self.api.upload_file(
                path_or_fileobj=config_path,
                path_in_repo=f"{self.model_name}_config.json",
                repo_id=self.repo_id,
                commit_message=f"Upload {self.model_name} configuration"
            )
            
            # Upload metadata
            metadata_path = model_path.replace('.pth', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            self.api.upload_file(
                path_or_fileobj=metadata_path,
                path_in_repo=f"{self.model_name}_metadata.json",
                repo_id=self.repo_id,
                commit_message=f"Upload {self.model_name} metadata"
            )
            
            # Clean up temporary files
            os.remove(config_path)
            os.remove(metadata_path)
            
            self.logger.info(f"✅ Model uploaded successfully to {self.repo_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to upload model: {e}")
            return False
    
    def upload_training_artifacts(
        self, 
        artifacts_dir: str,
        run_name: str
    ) -> bool:
        """
        Upload training artifacts (logs, plots, etc.) to Hugging Face.
        
        Args:
            artifacts_dir: Directory containing training artifacts
            run_name: Name of the training run
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(artifacts_dir):
                self.logger.warning(f"⚠️  Artifacts directory not found: {artifacts_dir}")
                return False
            
            # Create repository if it doesn't exist
            self.create_repository()
            
            # Upload all files in artifacts directory
            for file_path in Path(artifacts_dir).rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(artifacts_dir)
                    repo_path = f"artifacts/{run_name}/{relative_path}"
                    
                    self.api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=repo_path,
                        repo_id=self.repo_id,
                        commit_message=f"Upload training artifacts for {run_name}"
                    )
            
            self.logger.info(f"✅ Training artifacts uploaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to upload training artifacts: {e}")
            return False
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the uploaded model.
        
        Returns:
            Model information dictionary or None if failed
        """
        try:
            # Get repository info
            repo_info = self.api.repo_info(repo_id=self.repo_id)
            
            # Get model files
            files = self.api.list_repo_files(repo_id=self.repo_id)
            
            return {
                "repo_id": self.repo_id,
                "model_name": self.model_name,
                "files": files,
                "last_modified": repo_info.last_modified,
                "downloads": getattr(repo_info, 'downloads', 0)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get model info: {e}")
            return None

def upload_model_to_hf(
    model_path: str,
    model_config: Dict[str, Any],
    token: Optional[str] = None,
    repo_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Convenience function to upload a model to Hugging Face.
    
    Args:
        model_path: Path to the model file
        model_config: Model configuration
        token: Hugging Face token (optional, uses env var if not provided)
        repo_id: Repository ID (optional, uses env var if not provided)
        metadata: Additional metadata
        
    Returns:
        True if successful, False otherwise
    """
    try:
        hf_manager = HuggingFaceManager(token=token, repo_id=repo_id)
        return hf_manager.upload_model(model_path, model_config, metadata)
    except Exception as e:
        logging.error(f"❌ Failed to upload model to Hugging Face: {e}")
        return False

if __name__ == "__main__":
    # Test Hugging Face integration
    logging.basicConfig(level=logging.INFO)
    
    # Test with environment variables
    try:
        hf_manager = HuggingFaceManager()
        info = hf_manager.get_model_info()
        if info:
            print("✅ Hugging Face integration working")
            print(f"Repository: {info['repo_id']}")
            print(f"Files: {info['files']}")
        else:
            print("❌ Failed to get model info")
    except Exception as e:
        print(f"❌ Hugging Face integration failed: {e}")
        print("Make sure to set HUGGINGFACE_TOKEN and HUGGINGFACE_REPO_ID environment variables") 