# PDF Chat App - Backend

## Commands to run the app locally.

- "C:\Program Files\Python311\python.exe" -m venv virtual-env
    - To create a virtual env
    - use "where python" to know the python path
- .\virtual-env\Scripts\activate.bat
    - To enter into the virtual env
- pip install -r requirements.txt
    - To install the dependencies
- uvicorn main:app --reload
    - To Run the server

## Helpful resources:
- [How to enable CORS in FastAPI](https://fastapi.tiangolo.com/tutorial/cors/#more-info)