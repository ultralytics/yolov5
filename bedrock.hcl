version = "1.0"

train {
    step train {
        image = "basisai/workload-standard:v0.2.1"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements-train.txt",
        ]
        script = [{sh = ["python3 task_train.py"]}]
        resources {
            cpu = "2"
            memory = "14G"
            gpu = "1"
        }
    }

    parameters {
        BUCKET_NAME = "basisai-samples"
        DATA_DIR = "shellfish"
        EXECUTION_DATE = "2020-10-01"
        NUM_EPOCHS = "10"
    }
}

serve {
    image = "python:3.7"
    install = [
        "pip3 install --upgrade pip",
        "pip3 install -r requirements-serve.txt",
    ]
    script = [
        {sh = [
            "gunicorn --bind=:${BEDROCK_SERVER_PORT:-8080} --worker-class=gthread --workers=${WORKERS} --timeout=300 --preload serve_http:app"
        ]}
    ]

    parameters {
        WORKERS = "1"
    }
}
