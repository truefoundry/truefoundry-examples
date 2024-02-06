import os

def cached_download(cache_dir):
    def wrapper(fn):
        def internal(fqn):
            sanitized = fqn.replace(":", "--").replace("/", "-")
            download_dir = os.path.join(cache_dir, sanitized)
            os.makedirs(download_dir, exist_ok=True)
            done_path = os.path.join(download_dir, ".done")
            if not os.path.exists(done_path):
                final_path = fn(fqn, download_dir)
                with open(done_path, "w") as f:
                    f.write("")
                return final_path
            else:
                if fqn.startswith("artifact:"):
                    return os.path.join(download_dir, "files")
                elif fqn.startswith("model:"):
                    return os.path.join(download_dir, "files", "model")
                else:
                    raise NotImplementedError()
        return internal
    return wrapper