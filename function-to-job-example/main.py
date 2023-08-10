import typer
from module import normal, uniform

app = typer.Typer()

app.command()(normal)
app.command()(uniform)
app()