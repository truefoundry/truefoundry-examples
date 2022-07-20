import logging
import os

import torch
from fastapi.responses import HTMLResponse, JSONResponse
from servicefoundry.service import fastapi

logger = logging.getLogger(__name__)
app = fastapi.app()


@app.get(path="/test")
def test():
    gpu_available = torch.cuda.is_available()
    compute = None
    device = torch.device('cuda') if gpu_available else torch.device('cpu')
    a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
    b = torch.tensor([2, 3, 4], dtype=torch.float32, device=device)
    c = a + b
    compute = c.cpu().numpy().tolist()
    return {"gpu_available": gpu_available, "compute": compute}


@app.get("/", response_class=HTMLResponse)
def root():
    html_content = "<html><body>Open <a href='/docs'>Docs</a></body></html>"
    return HTMLResponse(content=html_content, status_code=200)

