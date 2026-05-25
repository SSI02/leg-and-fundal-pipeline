# Front End Wrapper

Start the wrapper from the `leg_pipeline` environment:

```bash
python "front end/app.py" --port 8070
```

Then open:

```text
http://127.0.0.1:8070
```

What it does:

- lets the user choose `leg` or `fundal`
- launches the existing manual scale picker inside the wrapper flow
- launches the existing anterior-frame picker for leg runs
- launches the existing seed picker for fundal runs
- starts the correct Python orchestrator and streams logs
- shows output buttons for summaries, JSON files, galleries, and the 3D viewer after completion
