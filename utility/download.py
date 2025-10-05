import os

def download_from_huggingface(model_id: str, local_path: str, subfolder: str = None, filename: str = None):
    """从Hugging Face下载模型"""
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        if filename:
            # 单文件下载
            file_path = subfolder + "/" + filename if subfolder else filename
            downloaded_file = hf_hub_download(
                repo_id=model_id,
                filename=file_path,
                local_dir=local_path,
                local_dir_use_symlinks=False
            )
            return downloaded_file
        else:
            # 全仓库下载
            if subfolder:
                snapshot_download(repo_id=model_id, local_dir=local_path, allow_patterns=[f"{subfolder}/*"])
            else:
                snapshot_download(repo_id=model_id, local_dir=local_path)
    except ImportError:
        raise Exception("huggingface_hub not installed. Please run: pip install huggingface_hub")


def download_from_modelscope(model_id: str, local_path: str, subfolder: str = None, filename: str = None):
    """从ModelScope下载模型"""
    cache_dir = os.path.join(local_path, "cache")
    try:
        from modelscope import snapshot_download
        if filename:
            # 单文件下载 - ModelScope目前没有直接的单文件下载API，使用snapshot_download + allow_patterns
            file_pattern = subfolder + "/" + filename if subfolder else filename
            downloaded_path = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                allow_patterns=[file_pattern]
            )
            # 返回下载的文件的完整路径
            downloaded_file = os.path.join(downloaded_path, file_pattern)
            return downloaded_file
        else:
            # 全仓库下载
            if subfolder:
                snapshot_download(model_id, local_dir=local_path, allow_patterns=[f"{subfolder}/*"])
            else:
                snapshot_download(model_id, local_dir=local_path,)
    except ImportError:
        raise Exception("modelscope not installed. Please run: pip install modelscope")
