from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import uuid

from .ising_model import IsingModel

app = FastAPI()


class SimulationState:
    def __init__(self):
        self.model: Optional[IsingModel] = None
        self.energy_history: List[float] = []
        self.magnetization_history: List[float] = []


# Store states in memory keyed by session_id
sessions: Dict[str, SimulationState] = {}


def get_session(x_session_id: str = Header(None)) -> SimulationState:
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header required")
    if x_session_id not in sessions:
        sessions[x_session_id] = SimulationState()
    return sessions[x_session_id]


class InitParams(BaseModel):
    N: int
    T: float


class StepParams(BaseModel):
    steps: int


class UpdateParams(BaseModel):
    T: float


@app.post("/api/session")
def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = SimulationState()
    return {"session_id": session_id}


@app.post("/api/init")
def init_model(params: InitParams, state: SimulationState = Depends(get_session)):
    state.model = IsingModel(params.N, params.T)
    state.energy_history = [state.model.cost()]
    state.magnetization_history = [state.model.magnetization()]
    return {"message": "Model initialized", "N": params.N, "T": params.T}


@app.post("/api/step")
def step(params: StepParams, state: SimulationState = Depends(get_session)):
    if state.model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")

    acceptance_rate, energy = state.model.run_mcmc(params.steps)
    mag = state.model.magnetization()

    state.energy_history.append(energy)
    state.magnetization_history.append(mag)

    return {"acceptance_rate": acceptance_rate, "energy": energy, "magnetization": mag}


@app.get("/api/state")
def get_state(state: SimulationState = Depends(get_session)):
    if state.model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")

    return {
        "config": state.model.config.tolist(),
        "energy": state.model.cost(),
        "magnetization": state.model.magnetization(),
        "T": state.model.T,
        "N": state.model.N,
    }


@app.get("/api/stats")
def get_stats(state: SimulationState = Depends(get_session)):
    if state.model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
        
    return {
        "energy": state.model.cost(),
        "magnetization": state.model.magnetization(),
        "T": state.model.T,
        "energy_history": state.energy_history,
        "magnetization_history": state.magnetization_history,
    }


@app.post("/api/params")
def update_params(params: UpdateParams, state: SimulationState = Depends(get_session)):
    if state.model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")

    state.model.T = params.T
    state.model.beta = 1.0 / params.T
    return {"message": "Parameters updated", "T": state.model.T}


static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_dir, "index.html"))
