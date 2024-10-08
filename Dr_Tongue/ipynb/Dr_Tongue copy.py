{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Installing libraries and Dependencies\n",
        "\n",
        "### Uncomment and run the following code block to install all required dependencies"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {},
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Collecting keras\n",
            "  Downloading keras-3.5.0-py3-none-any.whl.metadata (5.8 kB)\n",
            "Collecting absl-py (from keras)\n",
            "  Using cached absl_py-2.1.0-py3-none-any.whl.metadata (2.3 kB)\n",
            "Requirement already satisfied: numpy in /home/admsher/anaconda3/lib/python3.11/site-packages (from keras) (1.26.4)\n",
            "Requirement already satisfied: rich in /home/admsher/anaconda3/lib/python3.11/site-packages (from keras) (13.3.5)\n",
            "Collecting namex (from keras)\n",
            "  Using cached namex-0.0.8-py3-none-any.whl.metadata (246 bytes)\n",
            "Requirement already satisfied: h5py in /home/admsher/anaconda3/lib/python3.11/site-packages (from keras) (3.9.0)\n",
            "Collecting optree (from keras)\n",
            "  Downloading optree-0.12.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (47 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m47.8/47.8 kB\u001b[0m \u001b[31m739.7 kB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m \u001b[36m0:00:01\u001b[0m\n",
            "\u001b[?25hCollecting ml-dtypes (from keras)\n",
            "  Downloading ml_dtypes-0.5.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (21 kB)\n",
            "Requirement already satisfied: packaging in /home/admsher/anaconda3/lib/python3.11/site-packages (from keras) (23.1)\n",
            "Requirement already satisfied: typing-extensions>=4.5.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from optree->keras) (4.9.0)\n",
            "Requirement already satisfied: markdown-it-py<3.0.0,>=2.2.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from rich->keras) (2.2.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from rich->keras) (2.15.1)\n",
            "Requirement already satisfied: mdurl~=0.1 in /home/admsher/anaconda3/lib/python3.11/site-packages (from markdown-it-py<3.0.0,>=2.2.0->rich->keras) (0.1.0)\n",
            "Downloading keras-3.5.0-py3-none-any.whl (1.1 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m1.1/1.1 MB\u001b[0m \u001b[31m2.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m00:01\u001b[0m00:01\u001b[0m\n",
            "\u001b[?25hUsing cached absl_py-2.1.0-py3-none-any.whl (133 kB)\n",
            "Downloading ml_dtypes-0.5.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.5 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m4.5/4.5 MB\u001b[0m \u001b[31m2.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m00:01\u001b[0m00:01\u001b[0m\n",
            "\u001b[?25hUsing cached namex-0.0.8-py3-none-any.whl (5.8 kB)\n",
            "Downloading optree-0.12.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (349 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m349.1/349.1 kB\u001b[0m \u001b[31m2.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0ma \u001b[36m0:00:01\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: namex, optree, ml-dtypes, absl-py, keras\n",
            "Successfully installed absl-py-2.1.0 keras-3.5.0 ml-dtypes-0.5.0 namex-0.0.8 optree-0.12.1\n",
            "Requirement already satisfied: numpy in /home/admsher/anaconda3/lib/python3.11/site-packages (1.26.4)\n",
            "Collecting gradio\n",
            "  Downloading gradio-4.44.0-py3-none-any.whl.metadata (15 kB)\n",
            "Collecting aiofiles<24.0,>=22.0 (from gradio)\n",
            "  Downloading aiofiles-23.2.1-py3-none-any.whl.metadata (9.7 kB)\n",
            "Requirement already satisfied: anyio<5.0,>=3.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from gradio) (4.2.0)\n",
            "Collecting fastapi<1.0 (from gradio)\n",
            "  Downloading fastapi-0.115.0-py3-none-any.whl.metadata (27 kB)\n",
            "Collecting ffmpy (from gradio)\n",
            "  Downloading ffmpy-0.4.0-py3-none-any.whl.metadata (2.9 kB)\n",
            "Collecting gradio-client==1.3.0 (from gradio)\n",
            "  Downloading gradio_client-1.3.0-py3-none-any.whl.metadata (7.1 kB)\n",
            "Collecting httpx>=0.24.1 (from gradio)\n",
            "  Downloading httpx-0.27.2-py3-none-any.whl.metadata (7.1 kB)\n",
            "Collecting huggingface-hub>=0.19.3 (from gradio)\n",
            "  Downloading huggingface_hub-0.25.1-py3-none-any.whl.metadata (13 kB)\n",
            "Collecting importlib-resources<7.0,>=1.3 (from gradio)\n",
            "  Downloading importlib_resources-6.4.5-py3-none-any.whl.metadata (4.0 kB)\n",
            "Requirement already satisfied: jinja2<4.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from gradio) (3.1.3)\n",
            "Requirement already satisfied: markupsafe~=2.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from gradio) (2.1.3)\n",
            "Requirement already satisfied: matplotlib~=3.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from gradio) (3.8.0)\n",
            "Requirement already satisfied: numpy<3.0,>=1.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from gradio) (1.26.4)\n",
            "Collecting orjson~=3.0 (from gradio)\n",
            "  Downloading orjson-3.10.7-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (50 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m50.4/50.4 kB\u001b[0m \u001b[31m2.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: packaging in /home/admsher/anaconda3/lib/python3.11/site-packages (from gradio) (23.1)\n",
            "Requirement already satisfied: pandas<3.0,>=1.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from gradio) (2.1.4)\n",
            "Requirement already satisfied: pillow<11.0,>=8.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from gradio) (10.2.0)\n",
            "Collecting pydantic>=2.0 (from gradio)\n",
            "  Downloading pydantic-2.9.2-py3-none-any.whl.metadata (149 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m149.4/149.4 kB\u001b[0m \u001b[31m823.5 kB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0ma \u001b[36m0:00:01\u001b[0m\n",
            "\u001b[?25hCollecting pydub (from gradio)\n",
            "  Downloading pydub-0.25.1-py2.py3-none-any.whl.metadata (1.4 kB)\n",
            "Collecting python-multipart>=0.0.9 (from gradio)\n",
            "  Downloading python_multipart-0.0.10-py3-none-any.whl.metadata (1.9 kB)\n",
            "Requirement already satisfied: pyyaml<7.0,>=5.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from gradio) (6.0.1)\n",
            "Collecting ruff>=0.2.2 (from gradio)\n",
            "  Downloading ruff-0.6.7-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (25 kB)\n",
            "Requirement already satisfied: semantic-version~=2.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from gradio) (2.10.0)\n",
            "Collecting tomlkit==0.12.0 (from gradio)\n",
            "  Downloading tomlkit-0.12.0-py3-none-any.whl.metadata (2.7 kB)\n",
            "Collecting typer<1.0,>=0.12 (from gradio)\n",
            "  Downloading typer-0.12.5-py3-none-any.whl.metadata (15 kB)\n",
            "Requirement already satisfied: typing-extensions~=4.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from gradio) (4.9.0)\n",
            "Requirement already satisfied: urllib3~=2.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from gradio) (2.0.7)\n",
            "Requirement already satisfied: uvicorn>=0.14.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from gradio) (0.29.0)\n",
            "Requirement already satisfied: fsspec in /home/admsher/anaconda3/lib/python3.11/site-packages (from gradio-client==1.3.0->gradio) (2023.10.0)\n",
            "Collecting websockets<13.0,>=10.0 (from gradio-client==1.3.0->gradio)\n",
            "  Using cached websockets-12.0-cp311-cp311-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.6 kB)\n",
            "Requirement already satisfied: idna>=2.8 in /home/admsher/anaconda3/lib/python3.11/site-packages (from anyio<5.0,>=3.0->gradio) (3.4)\n",
            "Requirement already satisfied: sniffio>=1.1 in /home/admsher/anaconda3/lib/python3.11/site-packages (from anyio<5.0,>=3.0->gradio) (1.3.0)\n",
            "Requirement already satisfied: starlette<0.39.0,>=0.37.2 in /home/admsher/anaconda3/lib/python3.11/site-packages (from fastapi<1.0->gradio) (0.37.2)\n",
            "Requirement already satisfied: certifi in /home/admsher/anaconda3/lib/python3.11/site-packages (from httpx>=0.24.1->gradio) (2024.7.4)\n",
            "Collecting httpcore==1.* (from httpx>=0.24.1->gradio)\n",
            "  Using cached httpcore-1.0.5-py3-none-any.whl.metadata (20 kB)\n",
            "Requirement already satisfied: h11<0.15,>=0.13 in /home/admsher/anaconda3/lib/python3.11/site-packages (from httpcore==1.*->httpx>=0.24.1->gradio) (0.14.0)\n",
            "Requirement already satisfied: filelock in /home/admsher/anaconda3/lib/python3.11/site-packages (from huggingface-hub>=0.19.3->gradio) (3.13.1)\n",
            "Requirement already satisfied: requests in /home/admsher/anaconda3/lib/python3.11/site-packages (from huggingface-hub>=0.19.3->gradio) (2.31.0)\n",
            "Requirement already satisfied: tqdm>=4.42.1 in /home/admsher/anaconda3/lib/python3.11/site-packages (from huggingface-hub>=0.19.3->gradio) (4.65.0)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /home/admsher/anaconda3/lib/python3.11/site-packages (from matplotlib~=3.0->gradio) (1.2.0)\n",
            "Requirement already satisfied: cycler>=0.10 in /home/admsher/anaconda3/lib/python3.11/site-packages (from matplotlib~=3.0->gradio) (0.11.0)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from matplotlib~=3.0->gradio) (4.25.0)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in /home/admsher/anaconda3/lib/python3.11/site-packages (from matplotlib~=3.0->gradio) (1.4.4)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /home/admsher/anaconda3/lib/python3.11/site-packages (from matplotlib~=3.0->gradio) (3.0.9)\n",
            "Requirement already satisfied: python-dateutil>=2.7 in /home/admsher/anaconda3/lib/python3.11/site-packages (from matplotlib~=3.0->gradio) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /home/admsher/anaconda3/lib/python3.11/site-packages (from pandas<3.0,>=1.0->gradio) (2023.3.post1)\n",
            "Requirement already satisfied: tzdata>=2022.1 in /home/admsher/anaconda3/lib/python3.11/site-packages (from pandas<3.0,>=1.0->gradio) (2023.3)\n",
            "Collecting annotated-types>=0.6.0 (from pydantic>=2.0->gradio)\n",
            "  Using cached annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)\n",
            "Collecting pydantic-core==2.23.4 (from pydantic>=2.0->gradio)\n",
            "  Downloading pydantic_core-2.23.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.6 kB)\n",
            "Requirement already satisfied: click>=8.0.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from typer<1.0,>=0.12->gradio) (8.1.7)\n",
            "Collecting shellingham>=1.3.0 (from typer<1.0,>=0.12->gradio)\n",
            "  Using cached shellingham-1.5.4-py2.py3-none-any.whl.metadata (3.5 kB)\n",
            "Requirement already satisfied: rich>=10.11.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from typer<1.0,>=0.12->gradio) (13.3.5)\n",
            "Requirement already satisfied: six>=1.5 in /home/admsher/anaconda3/lib/python3.11/site-packages (from python-dateutil>=2.7->matplotlib~=3.0->gradio) (1.16.0)\n",
            "Requirement already satisfied: markdown-it-py<3.0.0,>=2.2.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from rich>=10.11.0->typer<1.0,>=0.12->gradio) (2.2.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from rich>=10.11.0->typer<1.0,>=0.12->gradio) (2.15.1)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /home/admsher/anaconda3/lib/python3.11/site-packages (from requests->huggingface-hub>=0.19.3->gradio) (2.0.4)\n",
            "Requirement already satisfied: mdurl~=0.1 in /home/admsher/anaconda3/lib/python3.11/site-packages (from markdown-it-py<3.0.0,>=2.2.0->rich>=10.11.0->typer<1.0,>=0.12->gradio) (0.1.0)\n",
            "Downloading gradio-4.44.0-py3-none-any.whl (18.1 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m18.1/18.1 MB\u001b[0m \u001b[31m477.9 kB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m00:01\u001b[0m00:01\u001b[0m\n",
            "\u001b[?25hDownloading gradio_client-1.3.0-py3-none-any.whl (318 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m318.7/318.7 kB\u001b[0m \u001b[31m519.9 kB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m00:01\u001b[0m00:01\u001b[0m\n",
            "\u001b[?25hDownloading tomlkit-0.12.0-py3-none-any.whl (37 kB)\n",
            "Downloading aiofiles-23.2.1-py3-none-any.whl (15 kB)\n",
            "Downloading fastapi-0.115.0-py3-none-any.whl (94 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m94.6/94.6 kB\u001b[0m \u001b[31m729.2 kB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0ma \u001b[36m0:00:01\u001b[0m\n",
            "\u001b[?25hDownloading httpx-0.27.2-py3-none-any.whl (76 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m76.4/76.4 kB\u001b[0m \u001b[31m697.6 kB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0ma \u001b[36m0:00:01\u001b[0m\n",
            "\u001b[?25hUsing cached httpcore-1.0.5-py3-none-any.whl (77 kB)\n",
            "Downloading huggingface_hub-0.25.1-py3-none-any.whl (436 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m436.4/436.4 kB\u001b[0m \u001b[31m528.3 kB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0ma \u001b[36m0:00:01\u001b[0m\n",
            "\u001b[?25hDownloading importlib_resources-6.4.5-py3-none-any.whl (36 kB)\n",
            "Downloading orjson-3.10.7-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (141 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m141.9/141.9 kB\u001b[0m \u001b[31m789.8 kB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0ma \u001b[36m0:00:01\u001b[0m\n",
            "\u001b[?25hDownloading pydantic-2.9.2-py3-none-any.whl (434 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m434.9/434.9 kB\u001b[0m \u001b[31m728.9 kB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0ma \u001b[36m0:00:01\u001b[0m\n",
            "\u001b[?25hDownloading pydantic_core-2.23.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m2.1/2.1 MB\u001b[0m \u001b[31m316.5 kB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m00:01\u001b[0m00:01\u001b[0m\n",
            "\u001b[?25hDownloading python_multipart-0.0.10-py3-none-any.whl (22 kB)\n",
            "Downloading ruff-0.6.7-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (10.8 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m10.8/10.8 MB\u001b[0m \u001b[31m566.4 kB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m00:01\u001b[0m00:01\u001b[0m\n",
            "\u001b[?25hDownloading typer-0.12.5-py3-none-any.whl (47 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m47.3/47.3 kB\u001b[0m \u001b[31m479.5 kB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0ma \u001b[36m0:00:01\u001b[0m\n",
            "\u001b[?25hDownloading ffmpy-0.4.0-py3-none-any.whl (5.8 kB)\n",
            "Downloading pydub-0.25.1-py2.py3-none-any.whl (32 kB)\n",
            "Using cached annotated_types-0.7.0-py3-none-any.whl (13 kB)\n",
            "Using cached shellingham-1.5.4-py2.py3-none-any.whl (9.8 kB)\n",
            "Using cached websockets-12.0-cp311-cp311-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (130 kB)\n",
            "Installing collected packages: pydub, websockets, tomlkit, shellingham, ruff, python-multipart, pydantic-core, orjson, importlib-resources, httpcore, ffmpy, annotated-types, aiofiles, pydantic, huggingface-hub, httpx, typer, gradio-client, fastapi, gradio\n",
            "  Attempting uninstall: tomlkit\n",
            "    Found existing installation: tomlkit 0.11.1\n",
            "    Uninstalling tomlkit-0.11.1:\n",
            "      Successfully uninstalled tomlkit-0.11.1\n",
            "  Attempting uninstall: pydantic\n",
            "    Found existing installation: pydantic 1.10.12\n",
            "    Uninstalling pydantic-1.10.12:\n",
            "      Successfully uninstalled pydantic-1.10.12\n",
            "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
            "anaconda-cloud-auth 0.1.4 requires pydantic<2.0, but you have pydantic 2.9.2 which is incompatible.\u001b[0m\u001b[31m\n",
            "\u001b[0mSuccessfully installed aiofiles-23.2.1 annotated-types-0.7.0 fastapi-0.115.0 ffmpy-0.4.0 gradio-4.44.0 gradio-client-1.3.0 httpcore-1.0.5 httpx-0.27.2 huggingface-hub-0.25.1 importlib-resources-6.4.5 orjson-3.10.7 pydantic-2.9.2 pydantic-core-2.23.4 pydub-0.25.1 python-multipart-0.0.10 ruff-0.6.7 shellingham-1.5.4 tomlkit-0.12.0 typer-0.12.5 websockets-12.0\n",
            "Collecting tensorflow\n",
            "  Downloading tensorflow-2.17.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.2 kB)\n",
            "Requirement already satisfied: absl-py>=1.0.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from tensorflow) (2.1.0)\n",
            "Collecting astunparse>=1.6.0 (from tensorflow)\n",
            "  Using cached astunparse-1.6.3-py2.py3-none-any.whl.metadata (4.4 kB)\n",
            "Collecting flatbuffers>=24.3.25 (from tensorflow)\n",
            "  Using cached flatbuffers-24.3.25-py2.py3-none-any.whl.metadata (850 bytes)\n",
            "Collecting gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 (from tensorflow)\n",
            "  Downloading gast-0.6.0-py3-none-any.whl.metadata (1.3 kB)\n",
            "Collecting google-pasta>=0.1.1 (from tensorflow)\n",
            "  Using cached google_pasta-0.2.0-py3-none-any.whl.metadata (814 bytes)\n",
            "Collecting h5py>=3.10.0 (from tensorflow)\n",
            "  Using cached h5py-3.11.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.5 kB)\n",
            "Collecting libclang>=13.0.0 (from tensorflow)\n",
            "  Using cached libclang-18.1.1-py2.py3-none-manylinux2010_x86_64.whl.metadata (5.2 kB)\n",
            "Collecting ml-dtypes<0.5.0,>=0.3.1 (from tensorflow)\n",
            "  Downloading ml_dtypes-0.4.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)\n",
            "Collecting opt-einsum>=2.3.2 (from tensorflow)\n",
            "  Using cached opt_einsum-3.3.0-py3-none-any.whl.metadata (6.5 kB)\n",
            "Requirement already satisfied: packaging in /home/admsher/anaconda3/lib/python3.11/site-packages (from tensorflow) (23.1)\n",
            "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /home/admsher/anaconda3/lib/python3.11/site-packages (from tensorflow) (3.20.3)\n",
            "Requirement already satisfied: requests<3,>=2.21.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from tensorflow) (2.31.0)\n",
            "Requirement already satisfied: setuptools in /home/admsher/anaconda3/lib/python3.11/site-packages (from tensorflow) (68.2.2)\n",
            "Requirement already satisfied: six>=1.12.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from tensorflow) (1.16.0)\n",
            "Collecting termcolor>=1.1.0 (from tensorflow)\n",
            "  Using cached termcolor-2.4.0-py3-none-any.whl.metadata (6.1 kB)\n",
            "Requirement already satisfied: typing-extensions>=3.6.6 in /home/admsher/anaconda3/lib/python3.11/site-packages (from tensorflow) (4.9.0)\n",
            "Requirement already satisfied: wrapt>=1.11.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from tensorflow) (1.14.1)\n",
            "Collecting grpcio<2.0,>=1.24.3 (from tensorflow)\n",
            "  Downloading grpcio-1.66.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.9 kB)\n",
            "Collecting tensorboard<2.18,>=2.17 (from tensorflow)\n",
            "  Downloading tensorboard-2.17.1-py3-none-any.whl.metadata (1.6 kB)\n",
            "Requirement already satisfied: keras>=3.2.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from tensorflow) (3.5.0)\n",
            "Collecting tensorflow-io-gcs-filesystem>=0.23.1 (from tensorflow)\n",
            "  Downloading tensorflow_io_gcs_filesystem-0.37.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (14 kB)\n",
            "Requirement already satisfied: numpy<2.0.0,>=1.23.5 in /home/admsher/anaconda3/lib/python3.11/site-packages (from tensorflow) (1.26.4)\n",
            "Requirement already satisfied: wheel<1.0,>=0.23.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from astunparse>=1.6.0->tensorflow) (0.43.0)\n",
            "Requirement already satisfied: rich in /home/admsher/anaconda3/lib/python3.11/site-packages (from keras>=3.2.0->tensorflow) (13.3.5)\n",
            "Requirement already satisfied: namex in /home/admsher/anaconda3/lib/python3.11/site-packages (from keras>=3.2.0->tensorflow) (0.0.8)\n",
            "Requirement already satisfied: optree in /home/admsher/anaconda3/lib/python3.11/site-packages (from keras>=3.2.0->tensorflow) (0.12.1)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /home/admsher/anaconda3/lib/python3.11/site-packages (from requests<3,>=2.21.0->tensorflow) (2.0.4)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /home/admsher/anaconda3/lib/python3.11/site-packages (from requests<3,>=2.21.0->tensorflow) (3.4)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /home/admsher/anaconda3/lib/python3.11/site-packages (from requests<3,>=2.21.0->tensorflow) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /home/admsher/anaconda3/lib/python3.11/site-packages (from requests<3,>=2.21.0->tensorflow) (2024.7.4)\n",
            "Requirement already satisfied: markdown>=2.6.8 in /home/admsher/anaconda3/lib/python3.11/site-packages (from tensorboard<2.18,>=2.17->tensorflow) (3.4.1)\n",
            "Collecting tensorboard-data-server<0.8.0,>=0.7.0 (from tensorboard<2.18,>=2.17->tensorflow)\n",
            "  Using cached tensorboard_data_server-0.7.2-py3-none-manylinux_2_31_x86_64.whl.metadata (1.1 kB)\n",
            "Requirement already satisfied: werkzeug>=1.0.1 in /home/admsher/anaconda3/lib/python3.11/site-packages (from tensorboard<2.18,>=2.17->tensorflow) (2.2.3)\n",
            "Requirement already satisfied: MarkupSafe>=2.1.1 in /home/admsher/anaconda3/lib/python3.11/site-packages (from werkzeug>=1.0.1->tensorboard<2.18,>=2.17->tensorflow) (2.1.3)\n",
            "Requirement already satisfied: markdown-it-py<3.0.0,>=2.2.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from rich->keras>=3.2.0->tensorflow) (2.2.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /home/admsher/anaconda3/lib/python3.11/site-packages (from rich->keras>=3.2.0->tensorflow) (2.15.1)\n",
            "Requirement already satisfied: mdurl~=0.1 in /home/admsher/anaconda3/lib/python3.11/site-packages (from markdown-it-py<3.0.0,>=2.2.0->rich->keras>=3.2.0->tensorflow) (0.1.0)\n",
            "Downloading tensorflow-2.17.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (601.3 MB)\n",
            "\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m\u001b[90m━━\u001b[0m \u001b[32m569.9/601.3 MB\u001b[0m \u001b[31m479.4 kB/s\u001b[0m eta \u001b[36m0:01:06\u001b[0m"
          ]
        }
      ],
      "source": [
        "!pip install keras\n",
        "!pip install numpy\n",
        "!pip install gradio\n",
        "!pip install tensorflow\n",
        "!pip install matplotlib\n",
        "!pip install scikit-learn \n",
        "!pip install opencv-python"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Importing libraries"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {},
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\numpy\\_distributor_init.py:30: UserWarning: loaded more than 1 DLL from .libs:\n",
            "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\numpy\\.libs\\libopenblas.PYQHXLVVQ7VESDPUVUADXEVJOBGHJPAY.gfortran-win_amd64.dll\n",
            "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\numpy\\.libs\\libopenblas.WCDJNK7YVMPZQ2ME2ZZHJJRJ3JIKNDB7.gfortran-win_amd64.dll\n",
            "  warnings.warn(\"loaded more than 1 DLL from .libs:\"\n"
          ]
        }
      ],
      "source": [
        "import cv2\n",
        "import os\n",
        "import numpy as np\n",
        "\n",
        "import gradio as gr\n",
        "\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "import tensorflow as tf\n",
        "from keras.optimizers import Adam\n",
        "from keras.models import Sequential\n",
        " \n",
        "from keras.preprocessing.image import ImageDataGenerator\n",
        "from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout\n",
        "\n",
        "from sklearn.metrics import classification_report"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Defining the `get_data` function to get data from the directory"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "id": "HO44eik8ms4p"
      },
      "outputs": [],
      "source": [
        "labels = ['red', 'black', 'geographic', 'normal', 'yellow'] # titles of subfolders\n",
        "img_size = 120 # input image size\n",
        "\n",
        "def get_data(data_dir):\n",
        "    data = [] \n",
        "    for label in labels: \n",
        "        path = os.path.join(data_dir, label)\n",
        "        class_num = labels.index(label)\n",
        "        for img in os.listdir(path):\n",
        "            try:\n",
        "                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format\n",
        "                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size\n",
        "                data.append([resized_arr, class_num])\n",
        "            except Exception as e:\n",
        "                print(e)\n",
        "    return np.array(data, dtype=object)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Defining training and validation directories"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 305
        },
        "id": "-r2x_dYEnA3Y",
        "outputId": "4c125064-f28a-49b1-e712-00f26f55b071"
      },
      "outputs": [],
      "source": [
        "train = get_data('/Dr. Tongue/data/train')\n",
        "val = get_data('/Dr. Tongue/data/test/')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Generating the features and labels from the data and normalizing it"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 235
        },
        "id": "XqVK--ygsGh6",
        "outputId": "d122c32e-7d07-4d43-defe-3b7ffd97474e"
      },
      "outputs": [],
      "source": [
        "x_train = []\n",
        "y_train = []\n",
        "x_val = []\n",
        "y_val = []\n",
        "\n",
        "for feature, label in train:\n",
        "  x_train.append(feature)\n",
        "  y_train.append(label)\n",
        "\n",
        "for feature, label in val:\n",
        "  x_val.append(feature)\n",
        "  y_val.append(label)\n",
        "\n",
        "# Normalize the data\n",
        "x_train = np.array(x_train) / 255\n",
        "x_val = np.array(x_val) / 255\n",
        "\n",
        "x_train.reshape(-1, img_size, img_size, 1)\n",
        "y_train = np.array(y_train)\n",
        "\n",
        "x_val.reshape(-1, img_size, img_size, 1)\n",
        "y_val = np.array(y_val)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Data Augmentation"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "id": "7rB8WNwhsRWU"
      },
      "outputs": [],
      "source": [
        "datagen = ImageDataGenerator(\n",
        "        featurewise_center=False,  # set input mean to 0 over the dataset\n",
        "        samplewise_center=False,  # set each sample mean to 0\n",
        "        featurewise_std_normalization=False,  # divide inputs by std of the dataset\n",
        "        samplewise_std_normalization=False,  # divide each input by its std\n",
        "        zca_whitening=False,  # apply ZCA whitening\n",
        "        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)\n",
        "        zoom_range = 0.2, # Randomly zoom image \n",
        "        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)\n",
        "        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)\n",
        "        horizontal_flip = True,  # randomly flip images\n",
        "        vertical_flip=False)  # randomly flip images\n",
        "\n",
        "datagen.fit(x_train)   "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Model Definition"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "id": "4BbPuhQwsU5Y"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Model: \"sequential\"\n",
            "_________________________________________________________________\n",
            "Layer (type)                 Output Shape              Param #   \n",
            "=================================================================\n",
            "conv2d (Conv2D)              (None, 120, 120, 32)      896       \n",
            "_________________________________________________________________\n",
            "max_pooling2d (MaxPooling2D) (None, 60, 60, 32)        0         \n",
            "_________________________________________________________________\n",
            "conv2d_1 (Conv2D)            (None, 60, 60, 32)        9248      \n",
            "_________________________________________________________________\n",
            "max_pooling2d_1 (MaxPooling2 (None, 30, 30, 32)        0         \n",
            "_________________________________________________________________\n",
            "conv2d_2 (Conv2D)            (None, 30, 30, 64)        18496     \n",
            "_________________________________________________________________\n",
            "max_pooling2d_2 (MaxPooling2 (None, 15, 15, 64)        0         \n",
            "_________________________________________________________________\n",
            "dropout (Dropout)            (None, 15, 15, 64)        0         \n",
            "_________________________________________________________________\n",
            "flatten (Flatten)            (None, 14400)             0         \n",
            "_________________________________________________________________\n",
            "dense (Dense)                (None, 128)               1843328   \n",
            "_________________________________________________________________\n",
            "dense_1 (Dense)              (None, 5)                 645       \n",
            "=================================================================\n",
            "Total params: 1,872,613\n",
            "Trainable params: 1,872,613\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ]
        }
      ],
      "source": [
        "model = Sequential()\n",
        "model.add(Conv2D(32,3,padding=\"same\", activation=\"relu\", input_shape=(120,120,3)))\n",
        "model.add(MaxPool2D())\n",
        "\n",
        "model.add(Conv2D(32, 3, padding=\"same\", activation=\"relu\"))\n",
        "model.add(MaxPool2D())\n",
        "\n",
        "model.add(Conv2D(64, 3, padding=\"same\", activation=\"relu\"))\n",
        "model.add(MaxPool2D())\n",
        "model.add(Dropout(0.4))\n",
        "\n",
        "model.add(Flatten())\n",
        "model.add(Dense(128,activation=\"relu\"))\n",
        "model.add(Dense(5, activation=\"softmax\"))\n",
        "\n",
        "model.summary()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Model Compilation"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "id": "5j2n06omsbmY"
      },
      "outputs": [],
      "source": [
        "opt = Adam(learning_rate=0.0005)\n",
        "model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Model fitting"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "id": "Xa0Q4KaRsiMB"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Epoch 1/25\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\keras\\backend.py:4929: UserWarning: \"`sparse_categorical_crossentropy` received `from_logits=True`, but the `output` argument was produced by a sigmoid or softmax activation and thus does not represent logits. Was this intended?\"\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "4/4 [==============================] - 37s 1s/step - loss: 1.6725 - accuracy: 0.1871 - val_loss: 1.5874 - val_accuracy: 0.2500\n",
            "Epoch 2/25\n",
            "4/4 [==============================] - 3s 779ms/step - loss: 1.5801 - accuracy: 0.3243 - val_loss: 1.5712 - val_accuracy: 0.4545\n",
            "Epoch 3/25\n",
            "4/4 [==============================] - 3s 741ms/step - loss: 1.5641 - accuracy: 0.4384 - val_loss: 1.5112 - val_accuracy: 0.5227\n",
            "Epoch 4/25\n",
            "4/4 [==============================] - 3s 750ms/step - loss: 1.4882 - accuracy: 0.5846 - val_loss: 1.4405 - val_accuracy: 0.5909\n",
            "Epoch 5/25\n",
            "4/4 [==============================] - 3s 743ms/step - loss: 1.3829 - accuracy: 0.7074 - val_loss: 1.2838 - val_accuracy: 0.6591\n",
            "Epoch 6/25\n",
            "4/4 [==============================] - 3s 735ms/step - loss: 1.2158 - accuracy: 0.6744 - val_loss: 1.1375 - val_accuracy: 0.5909\n",
            "Epoch 7/25\n",
            "4/4 [==============================] - 3s 749ms/step - loss: 1.0038 - accuracy: 0.7296 - val_loss: 0.9511 - val_accuracy: 0.7500\n",
            "Epoch 8/25\n",
            "4/4 [==============================] - 3s 741ms/step - loss: 0.8086 - accuracy: 0.7574 - val_loss: 0.8892 - val_accuracy: 0.6364\n",
            "Epoch 9/25\n",
            "4/4 [==============================] - 3s 790ms/step - loss: 0.6509 - accuracy: 0.8023 - val_loss: 0.9254 - val_accuracy: 0.5909\n",
            "Epoch 10/25\n",
            "4/4 [==============================] - 3s 896ms/step - loss: 0.5857 - accuracy: 0.8005 - val_loss: 0.8257 - val_accuracy: 0.7045\n",
            "Epoch 11/25\n",
            "4/4 [==============================] - 3s 676ms/step - loss: 0.5047 - accuracy: 0.8148 - val_loss: 1.1355 - val_accuracy: 0.5455\n",
            "Epoch 12/25\n",
            "4/4 [==============================] - 3s 682ms/step - loss: 0.4461 - accuracy: 0.8398 - val_loss: 1.0214 - val_accuracy: 0.6818\n",
            "Epoch 13/25\n",
            "4/4 [==============================] - 3s 693ms/step - loss: 0.5980 - accuracy: 0.8367 - val_loss: 0.9263 - val_accuracy: 0.5682\n",
            "Epoch 14/25\n",
            "4/4 [==============================] - 3s 690ms/step - loss: 0.3708 - accuracy: 0.8848 - val_loss: 0.7373 - val_accuracy: 0.7500\n",
            "Epoch 15/25\n",
            "4/4 [==============================] - 3s 690ms/step - loss: 0.3317 - accuracy: 0.8662 - val_loss: 0.7091 - val_accuracy: 0.8182\n",
            "Epoch 16/25\n",
            "4/4 [==============================] - 3s 805ms/step - loss: 0.3165 - accuracy: 0.9136 - val_loss: 0.7127 - val_accuracy: 0.6818\n",
            "Epoch 17/25\n",
            "4/4 [==============================] - 3s 732ms/step - loss: 0.2472 - accuracy: 0.9362 - val_loss: 0.7563 - val_accuracy: 0.7955\n",
            "Epoch 18/25\n",
            "4/4 [==============================] - 3s 696ms/step - loss: 0.2302 - accuracy: 0.9386 - val_loss: 0.7990 - val_accuracy: 0.6818\n",
            "Epoch 19/25\n",
            "4/4 [==============================] - 3s 709ms/step - loss: 0.1796 - accuracy: 0.9828 - val_loss: 0.5977 - val_accuracy: 0.7727\n",
            "Epoch 20/25\n",
            "4/4 [==============================] - 3s 707ms/step - loss: 0.1619 - accuracy: 0.9533 - val_loss: 0.7217 - val_accuracy: 0.7727\n",
            "Epoch 21/25\n",
            "4/4 [==============================] - 3s 677ms/step - loss: 0.1100 - accuracy: 0.9967 - val_loss: 0.7703 - val_accuracy: 0.7955\n",
            "Epoch 22/25\n",
            "4/4 [==============================] - 3s 703ms/step - loss: 0.0978 - accuracy: 0.9861 - val_loss: 0.6461 - val_accuracy: 0.7727\n",
            "Epoch 23/25\n",
            "4/4 [==============================] - 3s 677ms/step - loss: 0.0903 - accuracy: 0.9915 - val_loss: 0.7164 - val_accuracy: 0.7727\n",
            "Epoch 24/25\n",
            "4/4 [==============================] - 3s 685ms/step - loss: 0.0655 - accuracy: 1.0000 - val_loss: 0.7716 - val_accuracy: 0.7955\n",
            "Epoch 25/25\n",
            "4/4 [==============================] - 3s 672ms/step - loss: 0.0451 - accuracy: 1.0000 - val_loss: 0.6817 - val_accuracy: 0.7955\n"
          ]
        }
      ],
      "source": [
        "history = model.fit(x_train,y_train, epochs = 25, validation_data = (x_val, y_val))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Plotting Model's accuracy and loss w.r.t. training and validation set"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "fGJ0o8Fosr5a"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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",
            "text/plain": [
              "<Figure size 1440x720 with 2 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          },
          "output_type": "display_data"
        }
      ],
      "source": [
        "acc = history.history['accuracy']\n",
        "val_acc = history.history['val_accuracy']\n",
        "loss = history.history['loss']\n",
        "val_loss = history.history['val_loss']\n",
        "\n",
        "epochs_range = range(25)\n",
        "\n",
        "plt.figure(figsize=(20, 10))\n",
        "plt.subplot(2, 2, 1)\n",
        "plt.plot(epochs_range, acc, label='Training Accuracy')\n",
        "plt.plot(epochs_range, val_acc, label='Validation Accuracy')\n",
        "plt.legend(loc='lower right')\n",
        "plt.title('Training and Validation Accuracy')\n",
        "\n",
        "plt.subplot(2, 2, 2)\n",
        "plt.plot(epochs_range, loss, label='Training Loss')\n",
        "plt.plot(epochs_range, val_loss, label='Validation Loss')\n",
        "plt.legend(loc='upper right')\n",
        "plt.title('Training and Validation Loss')\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Making predictions"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "id": "jH-VxLOvuUqt"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\keras\\engine\\sequential.py:450: UserWarning: `model.predict_classes()` is deprecated and will be removed after 2021-01-01. Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype(\"int32\")`,   if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).\n",
            "  warnings.warn('`model.predict_classes()` is deprecated and '\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "                      precision    recall  f1-score   support\n",
            "\n",
            "       red (Class 0)       0.80      1.00      0.89         8\n",
            "     black (Class 1)       0.91      0.91      0.91        11\n",
            "geographic (Class 2)       1.00      0.44      0.62         9\n",
            "    normal (Class 3)       0.55      0.75      0.63         8\n",
            "    yellow (Class 4)       0.88      0.88      0.88         8\n",
            "\n",
            "            accuracy                           0.80        44\n",
            "           macro avg       0.83      0.80      0.78        44\n",
            "        weighted avg       0.84      0.80      0.79        44\n",
            "\n"
          ]
        }
      ],
      "source": [
        "predictions = model.predict_classes(x_val)\n",
        "predictions = predictions.reshape(1,-1)[0]\n",
        "print(classification_report(y_val, predictions, target_names = ['red (Class 0)','black (Class 1)', 'geographic (Class 2)', 'normal (Class 3)', 'yellow (Class 4)']))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Predicting using local web interface at `http://127.0.0.1:7860/`"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {},
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Running locally at: http://127.0.0.1:7860/\n",
            "To create a public link, set `share=True` in `launch()`.\n",
            "Interface loading below...\n"
          ]
        },
        {
          "data": {
            "text/html": [
              "\n",
              "        <iframe\n",
              "            width=\"900\"\n",
              "            height=\"500\"\n",
              "            src=\"http://127.0.0.1:7860/\"\n",
              "            frameborder=\"0\"\n",
              "            allowfullscreen\n",
              "        ></iframe>\n",
              "        "
            ],
            "text/plain": [
              "<IPython.lib.display.IFrame at 0x15b2f4d93a0>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "def predict_image(img):\n",
        "  img_4d=img.reshape(-1,120,120,3)\n",
        "  prediction=model.predict(img_4d)[0]\n",
        "  return {labels[i]: float(prediction[i]) for i in range(5)}\n",
        "\n",
        "image = gr.inputs.Image(shape=(120,120))\n",
        "label = gr.outputs.Label(num_top_classes=5)\n",
        "\n",
        "gr.Interface(fn=predict_image, inputs=image, outputs=label,interpretation='default').launch(debug='True')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## **Note:**\n",
        "\n",
        "### As can be observed, validation accuracy is lower than training accuracy, and validation loss is likewise higher than training loss. It's due to the fact that there's fewer data. By expanding the data set, accuracy may be improved and losses can be reduced."
      ]
    }
  ],
  "metadata": {
    "colab": {
      "collapsed_sections": [],
      "name": "Dr_Tongue.ipynb",
      "provenance": []
    },
    "interpreter": {
      "hash": "b3ba2566441a7c06988d0923437866b63cedc61552a5af99d1f4fb67d367b25f"
    },
    "kernelspec": {
      "display_name": "Python 3.8.5 64-bit ('base': conda)",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.11.7"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}
