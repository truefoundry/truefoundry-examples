import typer
from module1 import normal
from module2 import uniform

app = typer.Typer()

app.command()(normal)
app.command()(uniform)
app()