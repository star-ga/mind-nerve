"""
tests/bit_identity/corpus.py

Corpus assembler and writer for the A1.4 bit-identity harness.

Produces exactly 1,000 deterministic query strings:
  - 600 queries derived from the manifest eval_pairs distribution (route names
    and descriptions drawn from the live route_table, seeded for reproducibility).
  - 200 long queries (T > 256 tokens) — code snippets and multi-clause prompts
    that exercise the sliding-window encoder path.
  - 200 adversarial cases — all-unk tokens, leading whitespace, max-length
    boundary inputs, empty string.

Output: tests/bit_identity/corpus.json (JSON array of {"id", "text", "category"})
committed to repo so test runs are deterministic without a live runtime.

The corpus is independent of whether the native .so is built; it uses only the
route_table.jsonl names (which are committed data) plus the hardcoded seed
strings below.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Deterministic PRNG — xorshift64, same implementation as gen_fixtures.py.
# No numpy dependency in this file.
# ---------------------------------------------------------------------------

_CORPUS_SEED = 0xBEEF_CAFE_A1_04  # A1.4 corpus seed — NEVER change

CORPUS_SIZE = 1000
N_EVAL = 600
N_LONG = 200
N_ADVERSARIAL = 200

SLIDING_WINDOW = 256   # encoder max tokens per window
STRIDE = 192           # window stride (later-window-wins per §3.3)

THIS_DIR = Path(__file__).parent
CORPUS_PATH = THIS_DIR / "corpus.json"

# Runtime dir resolution — same env-var precedence as inference.py.
_DEFAULT_RUNTIME_DIR = Path.home() / ".local" / "share" / "mind-nerve" / "runtime"


class _Xorshift64:
    """Minimal xorshift64 PRNG. Deterministic, stdlib-only."""

    def __init__(self, seed: int) -> None:
        self._state = (seed & 0xFFFF_FFFF_FFFF_FFFF) or 1

    def next_u64(self) -> int:
        x = self._state
        x ^= (x << 13) & 0xFFFF_FFFF_FFFF_FFFF
        x ^= (x >> 7) & 0xFFFF_FFFF_FFFF_FFFF
        x ^= (x << 17) & 0xFFFF_FFFF_FFFF_FFFF
        self._state = x & 0xFFFF_FFFF_FFFF_FFFF
        return self._state

    def next_bounded(self, upper: int) -> int:
        """Uniform integer in [0, upper). Rejection-sampling to avoid bias."""
        assert upper > 0
        threshold = (1 << 64) - ((1 << 64) % upper)
        while True:
            v = self.next_u64()
            if v < threshold:
                return v % upper

    def shuffle(self, lst: list) -> list:
        """Fisher-Yates shuffle in-place. Returns the list."""
        n = len(lst)
        for i in range(n - 1, 0, -1):
            j = self.next_bounded(i + 1)
            lst[i], lst[j] = lst[j], lst[i]
        return lst


# ---------------------------------------------------------------------------
# Route table loader
# ---------------------------------------------------------------------------

def _route_names(runtime_dir: Path) -> list[str]:
    """Return all route names from route_table.jsonl."""
    jsonl = runtime_dir / "route_table.jsonl"
    if not jsonl.exists():
        return []
    names: list[str] = []
    with jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            name = obj.get("name", "").strip()
            if name:
                names.append(name)
    return names


def _resolve_runtime_dir() -> Path | None:
    env = os.environ.get("MIND_NERVE_RUNTIME_DIR")
    if env:
        p = Path(env).expanduser()
        return p if p.is_dir() else None
    return _DEFAULT_RUNTIME_DIR if _DEFAULT_RUNTIME_DIR.is_dir() else None


# ---------------------------------------------------------------------------
# Hard-coded 600-query seed — used when the runtime dir is not present.
# These are realistic CLI-agent query strings sampled from the eval distribution.
# ---------------------------------------------------------------------------

_HARDCODED_EVAL_SEED: list[str] = [
    "git status", "git diff HEAD", "git log --oneline -20",
    "git commit -m 'feat: add user auth'", "git push origin main",
    "git pull --rebase", "git stash", "git stash pop",
    "git checkout -b feature/new-api", "git merge main",
    "npm install", "npm run build", "npm test", "npm run dev",
    "pip install -r requirements.txt", "pip install -e '.[dev]'",
    "python -m pytest tests/ -v", "python -m pytest --cov",
    "docker build -t myapp .", "docker run -p 8080:80 myapp",
    "docker-compose up -d", "docker-compose down",
    "kubectl apply -f deployment.yaml", "kubectl get pods",
    "kubectl logs -f pod-name", "kubectl exec -it pod bash",
    "terraform plan", "terraform apply", "terraform destroy",
    "cargo build --release", "cargo test", "cargo clippy",
    "make build", "make test", "make clean",
    "ls -la", "find . -name '*.py' -type f",
    "grep -r 'TODO' src/", "grep -rn 'import' .",
    "cat README.md", "less CHANGELOG.md",
    "curl -X POST https://api.example.com/users -H 'Content-Type: application/json'",
    "curl -s https://api.example.com/health | jq .",
    "ssh user@host", "scp file.txt user@host:/tmp/",
    "chmod +x script.sh", "chown -R user:group /path",
    "top", "htop", "ps aux", "kill -9 1234",
    "systemctl status nginx", "systemctl restart nginx",
    "journalctl -f -u nginx", "tail -f /var/log/app.log",
    "ping google.com", "traceroute google.com",
    "netstat -tulpn", "ss -tlnp",
    "df -h", "du -sh *", "free -h",
    "tar -czf archive.tar.gz dir/", "tar -xzf archive.tar.gz",
    "zip -r archive.zip dir/", "unzip archive.zip",
    "openssl req -new -x509 -days 365 -keyout key.pem -out cert.pem",
    "export DATABASE_URL=postgres://localhost/mydb",
    "env | grep MIND", "printenv PATH",
    "which python3", "type git",
    "nmap -sV localhost", "tracepath example.com",
    "aws s3 ls s3://my-bucket", "aws ec2 describe-instances",
    "gcloud compute instances list", "gcloud auth login",
    "az vm list", "az group create --name mygroup --location eastus",
    "heroku logs --tail", "heroku config:set KEY=value",
    "flyctl deploy", "flyctl status",
    "pytest -x -v --tb=short", "pytest -k test_encoder",
    "ruff check src/", "ruff format --check src/",
    "mypy src/ --strict", "black src/ --check",
    "isort --check-only src/", "bandit -r src/",
    "eslint src/ --ext .ts", "tsc --noEmit",
    "vitest run", "jest --coverage",
    "gh pr create --title 'feat: encoder port'",
    "gh issue list", "gh repo clone org/repo",
    "jq '.data[] | .name' response.json",
    "yq '.spec.containers[0].image' pod.yaml",
    "sed -i 's/old/new/g' file.txt",
    "awk '{print $1, $3}' access.log",
    "sort -u names.txt", "uniq -c words.txt | sort -rn",
    "wc -l *.py", "head -100 large.log",
    "diff file1.txt file2.txt", "patch < changes.patch",
    "strace -p 1234", "ltrace ./binary",
    "valgrind --leak-check=full ./app",
    "gdb ./app core", "lldb ./app",
    "perf stat ./benchmark", "perf record -g ./app",
    "ab -n 1000 -c 10 http://localhost:8080/",
    "wrk -t4 -c100 -d30s http://localhost:8080/api/v1/",
    "siege -c 50 -r 100 http://localhost/",
    "locust -f locustfile.py --headless -u 100 -r 10",
    "k6 run script.js", "artillery run load.yml",
    "redis-cli ping", "redis-cli SET key value",
    "psql -U postgres -d mydb", "pg_dump mydb > backup.sql",
    "mysql -u root -p mydb", "mysqldump mydb > backup.sql",
    "mongosh --eval 'db.stats()'",
    "sqlite3 mydb.db '.tables'",
    "celery worker -A app.celery", "celery beat -A app.celery",
    "rabbitmq-server start", "kafka-topics --list",
    "ffmpeg -i input.mp4 -c:v h264 output.mp4",
    "convert image.jpg -resize 50% thumbnail.jpg",
    "optipng *.png", "jpegoptim *.jpg",
    "pandoc README.md -o README.pdf",
    "latex document.tex", "bibtex document",
    "node server.js", "node -e 'console.log(process.version)'",
    "deno run main.ts", "bun run start",
    "go build ./...", "go test ./...", "go mod tidy",
    "rustc main.rs", "rustup update",
    "mvn clean install", "mvn test",
    "gradle build", "gradle test",
    "dotnet build", "dotnet test",
    "php artisan serve", "php artisan migrate",
    "rails server", "rails db:migrate",
    "flask run --debug", "uvicorn main:app --reload",
    "gunicorn -w 4 main:app", "nginx -t",
    "apache2ctl configtest", "apache2ctl restart",
    "certbot renew", "certbot certonly --standalone",
    "fail2ban-client status", "ufw status numbered",
    "iptables -L -n", "nft list ruleset",
    "crontab -e", "at 2:00 AM tomorrow",
    "ansible-playbook site.yml -i inventory",
    "ansible all -m ping", "chef-client --local-mode",
    "puppet apply site.pp", "salt '*' test.ping",
    "vagrant up", "vagrant ssh", "packer build template.json",
    "helm install myapp ./chart", "helm upgrade myapp ./chart",
    "istioctl analyze", "kubectl rollout status deployment/myapp",
    "argocd app sync myapp", "flux reconcile source git flux-system",
    "prometheus --config.file=prometheus.yml",
    "grafana-server --config /etc/grafana/grafana.ini",
    "jaeger-all-in-one", "zipkin",
    "sentry-cli releases new 1.0.0",
    "datadog-agent status",
    "newrelic-admin run-program gunicorn main:app",
    "logstash -f logstash.conf", "kibana",
    "elasticsearch --daemon",
    "solr start -p 8983", "zookeeper-server-start.sh config/zookeeper.properties",
    "spark-submit --master local[4] job.py",
    "hadoop jar wordcount.jar WordCount input output",
    "hive -e 'SELECT COUNT(*) FROM my_table'",
    "pig -x local script.pig",
    "airflow dags list", "airflow tasks test my_dag task_id 2024-01-01",
    "dbt run", "dbt test", "dbt docs generate",
    "mlflow ui", "mlflow run . -P alpha=0.5",
    "bentoml serve service.py:svc",
    "triton-server --model-repository=/models",
    "ollama run llama2", "ollama pull mistral",
    "llamafile -m model.gguf --server",
    "litellm --model gpt-4 --port 8000",
    "openai api chat.completions.create -m gpt-4",
    "anthropic-sdk --version",
    "huggingface-cli download bert-base-uncased",
    "transformers-cli download --model bert-base-uncased",
    "optimum-cli export onnx --model bert-base-uncased bert_onnx",
    "onnxruntime --help",
    "tensorrt --help",
    "nvcc --version", "nvidia-smi", "nvtop",
    "python -c 'import torch; print(torch.cuda.is_available())'",
    "accelerate launch train.py",
    "deepspeed --num_gpus=4 train.py",
    "torchrun --nproc_per_node=4 train.py",
    "ray start --head", "ray up cluster.yaml",
    "dask-scheduler", "dask-worker scheduler:8786",
    "jupyter notebook", "jupyter lab",
    "voila notebook.ipynb", "nbconvert --to html notebook.ipynb",
    "papermill input.ipynb output.ipynb -p alpha 0.1",
    "snakemake --cores 4", "nextflow run pipeline.nf",
    "cwltool workflow.cwl input.yml",
    "cromwell run workflow.wdl inputs.json",
    "singularity exec myimage.sif python train.py",
    "apptainer build myimage.sif def.def",
    "lxc launch ubuntu:22.04 mycontainer",
    "qemu-system-x86_64 -m 4G -hda disk.img",
    "virsh start myvm", "virt-install --name myvm",
    "multipass launch 22.04 --name myvm",
    "lima start default.yaml",
    "nix-shell -p python3 numpy",
    "nix build .#myapp", "guix build python",
    "conda create -n myenv python=3.11",
    "conda activate myenv", "mamba install numpy",
    "poetry add requests", "poetry run python main.py",
    "pipx install black", "uv pip install requests",
    "pdm add requests", "flit build",
    "twine upload dist/*", "python -m build",
    "bump2version patch", "semantic-release version",
    "pre-commit run --all-files",
    "commitizen bump", "git-crypt unlock",
    "sops --decrypt secrets.enc.yaml",
    "vault kv get secret/myapp",
    "aws secretsmanager get-secret-value --secret-id myapp",
    "1password-cli read 'op://vault/item/field'",
    "bitwarden-cli get password myitem",
    "keybase sign -m 'message'",
    "gpg --armor --export user@example.com",
    "age -r recipient.pub plaintext.txt > encrypted.txt",
    "openssl enc -aes-256-cbc -in file -out file.enc",
    "checksec --file=./binary",
    "pwndbg ./vulnerable", "radare2 ./binary",
    "ghidra", "ida64",
    "objdump -d ./binary | head -50",
    "strings ./binary | grep -i password",
    "strace -e trace=network ./app",
    "tcpdump -i eth0 -w capture.pcap",
    "wireshark capture.pcap",
    "burpsuite", "zap-cli quick-scan http://localhost",
    "nikto -h http://localhost",
    "sqlmap -u 'http://localhost/user?id=1'",
    "nmap -sS -O localhost",
    "masscan --rate=1000 -p1-65535 192.168.1.0/24",
    "shodan search 'apache'",
    "metasploit", "msfconsole",
    "hydra -l admin -P passwords.txt http-get://localhost",
    "john --wordlist=rockyou.txt hashes.txt",
    "hashcat -a 0 -m 0 hashes.txt wordlist.txt",
    "aircrack-ng -w wordlist.txt -b 00:11:22:33:44:55 capture.cap",
    "steghide extract -sf image.jpg",
    "binwalk -e firmware.bin",
    "volatility -f memory.dmp --profile=Win7SP1x64 pslist",
    "autopsy", "sleuthkit fls -r image.dd",
    "foremost -i image.dd -o output/",
    "exiftool image.jpg", "exifprobe image.jpg",
    "mat2 --inplace document.pdf",
    "tshark -r capture.pcap -T fields -e http.request.uri",
    "suricata -c suricata.yaml -i eth0",
    "snort -c /etc/snort/snort.conf -i eth0",
    "zeek -r capture.pcap",
    "yara rules.yar suspicious.exe",
    "clamav --scan-dir=/tmp",
    "rkhunter --check", "chkrootkit",
    "lynis audit system", "openscap xccdf eval --profile xccdf_org",
    "trivy image myapp:latest", "grype myapp:latest",
    "syft myapp:latest -o spdx-json",
    "snyk test", "dependabot",
    "semgrep --config=p/python .",
    "sonarqube-scanner", "fortify sca -b myapp -scan",
    "checkmarx scan create", "veracode-api-signing",
    "prowler -g check11,check12 -r us-east-1",
    "steampipe query 'select * from aws_iam_user'",
    "cloudsplaining generate --input-file policy.json",
    "pacu modules list", "cloudmapper collect --account my-account",
    "trufflehog git --repo https://github.com/org/repo",
    "gitleaks detect --source .", "detect-secrets scan",
    "dockle myapp:latest", "hadolint Dockerfile",
    "kube-score score deployment.yaml",
    "polaris audit --config polaris.yaml --audit-path .",
    "kubeaudit all", "conftest test deployment.yaml",
    "opa eval --data policy.rego --input input.json",
    "terrascan scan -t aws -d .",
    "checkov -d .",
    "tflint --recursive",
    "regula run .",
    "cloud-custodian run -s out/ policy.yml",
    "falco --config /etc/falco/falco.yaml",
    "cilium status", "calico node status",
    "istio analyze", "linkerd check",
    "envoy --config-path envoy.yaml",
    "consul agent -dev", "nomad agent -dev",
    "vault server -dev", "boundary dev",
    "waypoint up", "packer init .",
    "pulumi up", "pulumi stack select dev",
    "cdk deploy", "cdk synth",
    "sam build", "sam deploy",
    "serverless deploy", "serverless invoke local",
    "netlify deploy --prod", "vercel --prod",
    "wrangler deploy", "pages-functions-build",
]


# Pad to 600 if needed (or trim to exactly 600)
def _ensure_600(base: list[str]) -> list[str]:
    rng = _Xorshift64(_CORPUS_SEED)
    result = list(base)
    # De-duplicate preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for s in result:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    result = deduped

    if len(result) > N_EVAL:
        result = result[:N_EVAL]
    elif len(result) < N_EVAL:
        # Pad with synthetic CLI commands
        extras = _synthetic_eval_queries(rng, N_EVAL - len(result), seen)
        result.extend(extras)
    return result


def _synthetic_eval_queries(rng: _Xorshift64, n: int, seen: set[str]) -> list[str]:
    templates = [
        "run {tool} --help",
        "list {tool} resources",
        "check {tool} status",
        "deploy {tool} to production",
        "build {tool} from source",
        "test {tool} connection",
        "configure {tool} settings",
        "update {tool} to latest",
        "install {tool} plugin",
        "export {tool} data",
    ]
    tools = [
        "terraform", "ansible", "kubernetes", "docker", "helm", "vault",
        "consul", "nomad", "packer", "vagrant", "jenkins", "gitlab",
        "github-actions", "circleci", "travisci", "drone", "argocd",
        "flux", "tekton", "knative", "istio", "linkerd", "envoy",
        "prometheus", "grafana", "alertmanager", "loki", "tempo",
        "jaeger", "zipkin", "opentelemetry", "datadog", "newrelic",
        "splunk", "elasticsearch", "kibana", "logstash", "fluentd",
    ]
    results: list[str] = []
    while len(results) < n:
        t = tools[rng.next_bounded(len(tools))]
        tmpl = templates[rng.next_bounded(len(templates))]
        q = tmpl.format(tool=t)
        if q not in seen:
            seen.add(q)
            results.append(q)
    return results


# ---------------------------------------------------------------------------
# Long queries (T > 256) — code snippets and multi-clause prompts.
# These are hardcoded strings that tokenize to > 256 tokens on BERT.
# ---------------------------------------------------------------------------

_LONG_QUERY_TEMPLATES: list[str] = [
    # Python code snippets
    (
        "def process_batch(items: list[dict], batch_size: int = 32, "
        "timeout: float = 30.0, retry_count: int = 3, "
        "backoff_factor: float = 1.5) -> list[dict]: "
        "\"\"\"Process a batch of items with retry logic and exponential backoff. "
        "Each item is validated, transformed, and persisted to the database. "
        "Failed items are retried up to retry_count times with exponential backoff. "
        "Items that fail all retries are logged and returned in the error list. "
        "The function is thread-safe and can be called from multiple workers. "
        "Parameters: items (list[dict]) — the items to process; "
        "batch_size (int) — number of items per database transaction; "
        "timeout (float) — per-item timeout in seconds; "
        "retry_count (int) — maximum retries per item; "
        "backoff_factor (float) — exponential backoff multiplier. "
        "Returns: list of successfully processed items.\"\"\" "
        "results = [] "
        "errors = [] "
        "for i in range(0, len(items), batch_size): "
        "    chunk = items[i:i + batch_size] "
        "    with db.transaction(): "
        "        for item in chunk: "
        "            for attempt in range(retry_count): "
        "                try: "
        "                    validated = validate(item) "
        "                    transformed = transform(validated) "
        "                    persisted = persist(transformed, timeout=timeout) "
        "                    results.append(persisted) "
        "                    break "
        "                except TransientError as e: "
        "                    if attempt == retry_count - 1: "
        "                        errors.append({'item': item, 'error': str(e)}) "
        "                    else: "
        "                        time.sleep(backoff_factor ** attempt) "
        "return results, errors"
    ),
    (
        "class DistributedEmbeddingCache: "
        "\"\"\"Thread-safe distributed embedding cache backed by Redis with "
        "LRU eviction, TTL support, and automatic serialization of "
        "numpy float32 arrays to Q16.16 fixed-point representation. "
        "The cache key is SHA-256(model_hash + tokenizer_hash + query_text). "
        "Cache entries expire after ttl_seconds (default 3600). "
        "The cache uses a two-level hierarchy: L1 is an in-process LRU "
        "(maxsize=1024) and L2 is the shared Redis cluster. "
        "Write-back from L1 to L2 is asynchronous via a background thread. "
        "The serialize method converts float32[384] to Q16.16 i32[384] bytes "
        "with deterministic rounding for cross-arch reproducibility.\"\"\" "
        "def __init__(self, redis_url: str, ttl_seconds: int = 3600, "
        "             l1_maxsize: int = 1024) -> None: "
        "    self._redis = redis.Redis.from_url(redis_url) "
        "    self._ttl = ttl_seconds "
        "    self._l1 = functools.lru_cache(maxsize=l1_maxsize)(self._get_l2) "
        "    self._write_queue: queue.Queue = queue.Queue(maxsize=4096) "
        "    self._writer = threading.Thread(target=self._write_worker, daemon=True) "
        "    self._writer.start() "
        "def _cache_key(self, model_hash: str, tokenizer_hash: str, query: str) -> str: "
        "    payload = f'{model_hash}:{tokenizer_hash}:{query}'.encode('utf-8') "
        "    return 'emb:' + hashlib.sha256(payload).hexdigest() "
        "def get(self, model_hash: str, tokenizer_hash: str, query: str) -> np.ndarray | None: "
        "    key = self._cache_key(model_hash, tokenizer_hash, query) "
        "    raw = self._redis.get(key) "
        "    if raw is None: "
        "        return None "
        "    return self._deserialize(raw) "
        "def put(self, model_hash: str, tokenizer_hash: str, query: str, "
        "        embedding: np.ndarray) -> None: "
        "    key = self._cache_key(model_hash, tokenizer_hash, query) "
        "    raw = self._serialize(embedding) "
        "    self._redis.setex(key, self._ttl, raw) "
        "def _serialize(self, vec: np.ndarray) -> bytes: "
        "    q16 = np.clip(np.round(vec * 65536).astype(np.int32), "
        "                  -2147483648, 2147483647) "
        "    return q16.tobytes() "
        "def _deserialize(self, raw: bytes) -> np.ndarray: "
        "    q16 = np.frombuffer(raw, dtype=np.int32) "
        "    return (q16.astype(np.float32) / 65536.0)"
    ),
    # Multi-clause architecture prompts
    (
        "Analyze the following microservices architecture and identify potential "
        "single points of failure, bottlenecks, and security vulnerabilities: "
        "The system consists of an API gateway that routes requests to twelve "
        "backend services including user-service, auth-service, payment-service, "
        "notification-service, email-service, sms-service, recommendation-service, "
        "search-service, catalog-service, inventory-service, order-service, and "
        "shipping-service. Each service has its own PostgreSQL database with "
        "connection pooling configured via PgBouncer. Services communicate "
        "asynchronously via RabbitMQ for non-critical paths and synchronously "
        "via gRPC for critical paths like payment processing. The API gateway "
        "is implemented in Nginx with Lua scripting for rate limiting. "
        "Authentication uses JWT tokens with a 15-minute expiry refreshed "
        "automatically by the client. The recommendation service uses a "
        "pre-trained collaborative filtering model served via TorchServe. "
        "All services are deployed as Kubernetes pods with horizontal pod "
        "autoscaling configured based on CPU and custom Prometheus metrics. "
        "The system handles approximately 50,000 requests per second at peak "
        "with a p99 latency SLO of 500ms for API gateway responses."
    ),
    (
        "Implement a complete CI/CD pipeline for a Python machine learning project "
        "that includes the following stages: code quality checks using ruff, black, "
        "isort, mypy, and bandit; unit tests with pytest and coverage reporting; "
        "integration tests against a real PostgreSQL database using testcontainers; "
        "model training on a held-out validation set with automatic hyperparameter "
        "tuning using Optuna; model evaluation comparing the new model against the "
        "currently deployed baseline on a standardized benchmark dataset of 10,000 "
        "samples; model registration in MLflow with artifact storage on S3; "
        "Docker image build with multi-stage build optimized for production size "
        "under 500MB; security scanning using trivy and snyk; deployment to "
        "Kubernetes staging environment with smoke tests; automated load testing "
        "using k6 to validate p99 latency under 200ms at 1000 concurrent users; "
        "canary deployment to production with 5% traffic split; monitoring of "
        "key metrics for 30 minutes before full rollout; automatic rollback if "
        "error rate exceeds 0.5% or p99 latency exceeds 500ms during canary phase. "
        "The pipeline should run on GitHub Actions with separate jobs for each "
        "stage, using job dependencies to ensure correct execution order. "
        "Secrets should be managed via GitHub OIDC authentication to AWS without "
        "storing long-lived credentials."
    ),
    # TypeScript code snippets
    (
        "import { createServer } from 'http'; "
        "import { WebSocketServer, WebSocket } from 'ws'; "
        "import { EventEmitter } from 'events'; "
        "interface ClientMessage { type: 'subscribe' | 'unsubscribe' | 'publish'; "
        "  channel: string; payload?: unknown; clientId: string; } "
        "interface ServerMessage { type: 'data' | 'error' | 'ack'; "
        "  channel?: string; payload?: unknown; timestamp: number; } "
        "class PubSubServer extends EventEmitter { "
        "  private channels = new Map<string, Set<WebSocket>>(); "
        "  private clientChannels = new Map<WebSocket, Set<string>>(); "
        "  constructor(private readonly port: number) { super(); } "
        "  subscribe(ws: WebSocket, channel: string): void { "
        "    if (!this.channels.has(channel)) this.channels.set(channel, new Set()); "
        "    this.channels.get(channel)!.add(ws); "
        "    if (!this.clientChannels.has(ws)) this.clientChannels.set(ws, new Set()); "
        "    this.clientChannels.get(ws)!.add(channel); } "
        "  unsubscribe(ws: WebSocket, channel: string): void { "
        "    this.channels.get(channel)?.delete(ws); "
        "    this.clientChannels.get(ws)?.delete(channel); } "
        "  publish(channel: string, payload: unknown): void { "
        "    const msg: ServerMessage = { type: 'data', channel, payload, "
        "      timestamp: Date.now() }; "
        "    const data = JSON.stringify(msg); "
        "    this.channels.get(channel)?.forEach(client => { "
        "      if (client.readyState === WebSocket.OPEN) client.send(data); }); } "
        "  cleanup(ws: WebSocket): void { "
        "    this.clientChannels.get(ws)?.forEach(ch => this.unsubscribe(ws, ch)); "
        "    this.clientChannels.delete(ws); } "
        "  listen(): void { "
        "    const server = createServer(); "
        "    const wss = new WebSocketServer({ server }); "
        "    wss.on('connection', ws => { "
        "      ws.on('message', raw => { "
        "        const msg: ClientMessage = JSON.parse(raw.toString()); "
        "        switch (msg.type) { "
        "          case 'subscribe': this.subscribe(ws, msg.channel); break; "
        "          case 'unsubscribe': this.unsubscribe(ws, msg.channel); break; "
        "          case 'publish': this.publish(msg.channel, msg.payload); break; } }); "
        "      ws.on('close', () => this.cleanup(ws)); }); "
        "    server.listen(this.port); } }"
    ),
    # Rust code snippet
    (
        "use std::collections::HashMap; "
        "use std::sync::{Arc, RwLock}; "
        "use std::time::{Duration, Instant}; "
        "#[derive(Debug, Clone)] "
        "pub struct CacheEntry<V> { value: V, inserted_at: Instant, ttl: Duration, } "
        "impl<V> CacheEntry<V> { "
        "    pub fn is_expired(&self) -> bool { "
        "        self.inserted_at.elapsed() >= self.ttl } } "
        "#[derive(Debug)] "
        "pub struct TtlCache<K, V> { "
        "    inner: Arc<RwLock<HashMap<K, CacheEntry<V>>>>, "
        "    default_ttl: Duration, } "
        "impl<K, V> TtlCache<K, V> "
        "where K: Eq + std::hash::Hash + Clone, V: Clone, { "
        "    pub fn new(default_ttl: Duration) -> Self { "
        "        Self { inner: Arc::new(RwLock::new(HashMap::new())), default_ttl } } "
        "    pub fn insert(&self, key: K, value: V) { "
        "        let entry = CacheEntry { value, inserted_at: Instant::now(), "
        "            ttl: self.default_ttl }; "
        "        self.inner.write().unwrap().insert(key, entry); } "
        "    pub fn get(&self, key: &K) -> Option<V> { "
        "        let guard = self.inner.read().unwrap(); "
        "        guard.get(key).filter(|e| !e.is_expired()).map(|e| e.value.clone()) } "
        "    pub fn evict_expired(&self) -> usize { "
        "        let mut guard = self.inner.write().unwrap(); "
        "        let before = guard.len(); "
        "        guard.retain(|_, e| !e.is_expired()); "
        "        before - guard.len() } "
        "    pub fn len(&self) -> usize { self.inner.read().unwrap().len() } "
        "    pub fn is_empty(&self) -> bool { self.len() == 0 } }"
    ),
    # SQL and database queries
    (
        "Write a PostgreSQL query to find all users who have placed more than 5 orders "
        "in the last 30 days, where each order total exceeds $100, and who have not "
        "been flagged for fraud, grouped by user_id with aggregated statistics including "
        "total order count, total spend, average order value, first order date, last "
        "order date, and the names of the top 3 most frequently ordered products for "
        "each user, sorted by total spend descending, with pagination support using "
        "cursor-based pagination on user_id for efficient handling of large result sets. "
        "The query should use CTEs for readability, include appropriate indexes on the "
        "orders table for (user_id, created_at, status) and a partial index on "
        "(user_id) WHERE status = 'completed', and should complete in under 100ms "
        "on a table with 10 million rows using PostgreSQL 16 with the pg_stat_statements "
        "extension enabled for query performance monitoring."
    ),
    # Infrastructure as code
    (
        "Create a Terraform configuration for a highly available three-tier web "
        "application on AWS consisting of: a public-facing Application Load Balancer "
        "with WAF integration and DDoS protection via AWS Shield Standard; an Auto "
        "Scaling Group of EC2 t3.medium instances running in private subnets across "
        "three availability zones with a minimum of 3 and maximum of 20 instances; "
        "an RDS PostgreSQL 16 Multi-AZ cluster with read replicas in each availability "
        "zone, automated backups retained for 30 days, and encryption at rest using "
        "AWS KMS; ElastiCache Redis 7 cluster mode enabled with 3 shards and 2 "
        "replicas per shard for session storage and caching; an S3 bucket with "
        "versioning, cross-region replication to us-west-2, and lifecycle policies "
        "to transition objects to Glacier after 90 days; CloudFront distribution "
        "fronting the ALB with custom SSL certificate from ACM, security headers "
        "policy, and edge caching for static assets; VPC with CIDR 10.0.0.0/16 "
        "split into 6 subnets (3 public /24, 3 private /24) with NAT gateways in "
        "each AZ; Security Groups following principle of least privilege with "
        "separate groups for ALB, web tier, app tier, and database tier."
    ),
    # Architecture decision record
    (
        "Architectural Decision Record: Switching from synchronous REST API calls "
        "to an event-driven architecture using Apache Kafka for the order processing "
        "pipeline. Context: The current synchronous REST-based order processing "
        "system suffers from cascading failures when downstream services are slow "
        "or unavailable. Peak load of 10,000 orders per minute causes timeouts in "
        "the payment service which propagates back to the API gateway causing "
        "HTTP 503 errors visible to customers. The average order processing time "
        "is 2.3 seconds due to sequential API calls to inventory, payment, "
        "notification, and shipping services. Decision: Implement event-driven "
        "architecture using Apache Kafka as the message broker. Each service "
        "publishes domain events and subscribes to relevant topics. Order creation "
        "publishes to order.created topic. Payment service subscribes to order.created "
        "and publishes payment.processed or payment.failed. Inventory service "
        "subscribes to payment.processed and publishes inventory.reserved or "
        "inventory.insufficient. The API immediately returns 202 Accepted with "
        "an order ID. Customers poll a read model or receive webhook notifications "
        "when order status changes. Consequences: Orders will be processed "
        "asynchronously within 5 seconds p99. Individual service failures will "
        "not cause customer-visible errors. We gain horizontal scalability for "
        "each consumer group independently. We introduce eventual consistency "
        "which requires UI changes to handle pending order states gracefully."
    ),
    # Security review
    (
        "Conduct a comprehensive security review of the following authentication "
        "implementation and identify all vulnerabilities with CVSS scores: "
        "The system uses JWT tokens with HS256 signing using a hardcoded secret "
        "key 'super_secret_key_12345' stored directly in the application code. "
        "Tokens have a 30-day expiry with no refresh mechanism. The token payload "
        "includes the user_id, role, email, and a boolean is_admin flag that is "
        "trusted by all downstream services without verification. Password hashing "
        "uses MD5 with a 4-character salt. The login endpoint does not implement "
        "rate limiting. Session tokens are stored in localStorage instead of "
        "httpOnly cookies. The password reset flow sends a 6-digit numeric OTP "
        "via email with a 24-hour expiry and no attempt limiting. The API does "
        "not validate Content-Type headers on POST requests. SQL queries for "
        "user lookup use string concatenation without parameterization. "
        "Cross-origin requests are allowed from all origins via Access-Control-Allow-Origin: *. "
        "The application runs as root in production containers. Debug endpoints "
        "exposing stack traces and environment variables are accessible without "
        "authentication in production. Please provide remediation recommendations "
        "for each vulnerability ordered by severity."
    ),
    # Performance optimization
    (
        "Optimize the following Python function that processes a list of 100,000 "
        "user records to compute personalized recommendations: "
        "def compute_recommendations(users: list[dict], products: list[dict], "
        "interactions: list[dict]) -> dict[str, list[str]]: "
        "    results = {} "
        "    for user in users: "
        "        user_interactions = [i for i in interactions "
        "                             if i['user_id'] == user['id']] "
        "        interacted_products = {i['product_id'] for i in user_interactions} "
        "        similar_users = [] "
        "        for other_user in users: "
        "            if other_user['id'] == user['id']: continue "
        "            other_interactions = {i['product_id'] for i in interactions "
        "                                  if i['user_id'] == other_user['id']} "
        "            if len(interacted_products & other_interactions) >= 3: "
        "                similar_users.append(other_user['id']) "
        "        recommended = [] "
        "        for similar_user_id in similar_users[:10]: "
        "            similar_interactions = {i['product_id'] for i in interactions "
        "                                    if i['user_id'] == similar_user_id} "
        "            for prod_id in similar_interactions - interacted_products: "
        "                if prod_id not in recommended: "
        "                    recommended.append(prod_id) "
        "        results[user['id']] = recommended[:20] "
        "    return results "
        "The current implementation has O(n^3) complexity. Propose and implement "
        "an optimized version using numpy vectorized operations, sparse matrix "
        "representations, and approximate nearest neighbor search that reduces "
        "wall-clock time from 45 minutes to under 30 seconds on a 16-core machine."
    ),
    # Kubernetes manifest
    (
        "apiVersion: apps/v1 "
        "kind: Deployment "
        "metadata: "
        "  name: mind-nerve-encoder "
        "  namespace: production "
        "  labels: "
        "    app: mind-nerve "
        "    component: encoder "
        "    version: v0.4.0 "
        "spec: "
        "  replicas: 3 "
        "  selector: "
        "    matchLabels: "
        "      app: mind-nerve "
        "      component: encoder "
        "  strategy: "
        "    type: RollingUpdate "
        "    rollingUpdate: "
        "      maxSurge: 1 "
        "      maxUnavailable: 0 "
        "  template: "
        "    metadata: "
        "      labels: "
        "        app: mind-nerve "
        "        component: encoder "
        "        version: v0.4.0 "
        "      annotations: "
        "        prometheus.io/scrape: 'true' "
        "        prometheus.io/port: '9090' "
        "        prometheus.io/path: '/metrics' "
        "    spec: "
        "      securityContext: "
        "        runAsNonRoot: true "
        "        runAsUser: 1000 "
        "        fsGroup: 1000 "
        "      containers: "
        "      - name: encoder "
        "        image: ghcr.io/star-ga/mind-nerve:v0.4.0 "
        "        ports: "
        "        - containerPort: 8080 "
        "        - containerPort: 9090 "
        "        resources: "
        "          requests: "
        "            cpu: 500m "
        "            memory: 512Mi "
        "          limits: "
        "            cpu: 2000m "
        "            memory: 2Gi "
        "        readinessProbe: "
        "          httpGet: "
        "            path: /health/ready "
        "            port: 8080 "
        "          initialDelaySeconds: 10 "
        "          periodSeconds: 5 "
        "        livenessProbe: "
        "          httpGet: "
        "            path: /health/live "
        "            port: 8080 "
        "          initialDelaySeconds: 30 "
        "          periodSeconds: 10 "
        "        env: "
        "        - name: MIND_NERVE_BACKEND "
        "          value: native "
        "        - name: MIND_NERVE_RUNTIME_DIR "
        "          value: /runtime "
        "        volumeMounts: "
        "        - name: runtime "
        "          mountPath: /runtime "
        "          readOnly: true "
        "      volumes: "
        "      - name: runtime "
        "        configMap: "
        "          name: mind-nerve-runtime"
    ),
    # OpenAPI spec fragment
    (
        "openapi: 3.1.0 "
        "info: "
        "  title: mind-nerve Encoding API "
        "  version: 0.4.0 "
        "  description: | "
        "    Q16.16 native BERT encoder for intent classification and tool routing. "
        "    Encodes query strings into 384-dimensional L2-normalized embeddings "
        "    and returns top-K routing candidates from a precomputed catalog. "
        "    The encode endpoint accepts a JSON body with a query string and "
        "    optional parameters for top_k, backend selection, and runtime override. "
        "    All embeddings are returned as Q16.16 fixed-point integers to ensure "
        "    bit-identical results across x86_64 CPU backends. CUDA and ARM backends "
        "    are available but require explicit opt-in via the backend parameter. "
        "    Rate limiting is enforced at 1000 requests per minute per API key. "
        "    The catalog endpoint returns metadata about the currently loaded "
        "    routing catalog including version, size, and hash for cache validation. "
        "paths: "
        "  /v1/encode: "
        "    post: "
        "      operationId: encodeQuery "
        "      summary: Encode a query string and return top-K routing candidates "
        "      requestBody: "
        "        required: true "
        "        content: "
        "          application/json: "
        "            schema: "
        "              type: object "
        "              required: [query] "
        "              properties: "
        "                query: "
        "                  type: string "
        "                  minLength: 1 "
        "                  maxLength: 2048 "
        "                  description: The query string to encode and route "
        "                top_k: "
        "                  type: integer "
        "                  minimum: 1 "
        "                  maximum: 100 "
        "                  default: 5 "
        "                backend: "
        "                  type: string "
        "                  enum: [native, pytorch, cuda] "
        "                  default: native "
        "      responses: "
        "        '200': "
        "          description: Successful encoding and routing "
        "          content: "
        "            application/json: "
        "              schema: "
        "                $ref: '#/components/schemas/RouteResult' "
        "        '400': "
        "          description: Invalid request (query too long, invalid backend) "
        "        '429': "
        "          description: Rate limit exceeded "
        "        '503': "
        "          description: Backend unavailable"
    ),
    # Shell script
    (
        "#!/usr/bin/env bash "
        "set -euo pipefail "
        "IFS=$'\\n\\t' "
        "# Automated deployment script for mind-nerve v0.4.0 "
        "# Performs: build, test, security scan, docker build, push, deploy "
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\" "
        "REPO_ROOT=\"$(cd \"${SCRIPT_DIR}/..\" && pwd)\" "
        "VERSION=\"${1:-}\" "
        "ENVIRONMENT=\"${2:-staging}\" "
        "REGISTRY=\"ghcr.io/star-ga\" "
        "IMAGE=\"${REGISTRY}/mind-nerve\" "
        "if [[ -z \"${VERSION}\" ]]; then "
        "    echo 'ERROR: VERSION argument required' >&2 "
        "    echo 'Usage: deploy.sh <version> [environment]' >&2 "
        "    exit 1 "
        "fi "
        "echo \"==> Deploying mind-nerve ${VERSION} to ${ENVIRONMENT}\" "
        "cd \"${REPO_ROOT}\" "
        "echo '==> Running tests...' "
        "python -m pytest tests/ -x -q --tb=short "
        "echo '==> Running security scan...' "
        "bandit -r python/mind_nerve -ll "
        "echo '==> Building Docker image...' "
        "docker build --build-arg VERSION=\"${VERSION}\" "
        "             --build-arg BUILD_DATE=\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\" "
        "             --build-arg VCS_REF=\"$(git rev-parse HEAD)\" "
        "             -t \"${IMAGE}:${VERSION}\" "
        "             -t \"${IMAGE}:latest\" "
        "             . "
        "echo '==> Scanning image for vulnerabilities...' "
        "trivy image --exit-code 1 --severity HIGH,CRITICAL \"${IMAGE}:${VERSION}\" "
        "echo '==> Pushing image...' "
        "docker push \"${IMAGE}:${VERSION}\" "
        "docker push \"${IMAGE}:latest\" "
        "echo '==> Deploying to Kubernetes...' "
        "kubectl set image deployment/mind-nerve-encoder "
        "        encoder=\"${IMAGE}:${VERSION}\" "
        "        --namespace=\"${ENVIRONMENT}\" "
        "kubectl rollout status deployment/mind-nerve-encoder "
        "        --namespace=\"${ENVIRONMENT}\" "
        "        --timeout=300s "
        "echo \"==> Deployment complete: ${IMAGE}:${VERSION} -> ${ENVIRONMENT}\" "
    ),
    # Data pipeline
    (
        "Design and implement a fault-tolerant data pipeline that ingests "
        "streaming events from Apache Kafka topics (user.clicked, user.purchased, "
        "user.viewed, user.searched) at a rate of 500,000 events per second, "
        "applies real-time feature engineering including session reconstruction "
        "with a 30-minute session timeout, rolling window aggregations over "
        "1-minute, 5-minute, and 1-hour windows, user-level and item-level "
        "statistics, and categorical feature encoding using frequency-adaptive "
        "target encoding with Laplace smoothing; writes the enriched feature "
        "vectors to a Redis Feature Store with a 24-hour TTL for online serving "
        "and to Apache Iceberg tables on S3 for offline training; handles late "
        "arriving events up to 5 minutes behind watermark using Apache Flink's "
        "event-time processing with allowed lateness; implements exactly-once "
        "semantics via Kafka transactions and Flink checkpointing to S3 every "
        "60 seconds; provides a Grafana dashboard showing end-to-end latency "
        "p50/p95/p99, throughput, checkpoint success rate, and per-topic lag; "
        "auto-scales Flink TaskManagers from 4 to 64 based on Kafka consumer "
        "lag using KEDA; and recovers automatically from TaskManager failures "
        "within 30 seconds by restoring from the latest checkpoint."
    ),
    # Go implementation
    (
        "package encoder "
        "import ( "
        "    'context' "
        "    'crypto/sha256' "
        "    'encoding/binary' "
        "    'fmt' "
        "    'math' "
        "    'sync' "
        ") "
        "const ( "
        "    HiddenDim   = 384 "
        "    NumHeads    = 12 "
        "    HeadDim     = 32 "
        "    Q16Scale    = 65536.0 "
        "    LayerNormEps = 1e-12 "
        ") "
        "type Q16 int32 "
        "func Float32ToQ16(f float32) Q16 { "
        "    scaled := float64(f) * Q16Scale "
        "    if scaled > math.MaxInt32 { return Q16(math.MaxInt32) } "
        "    if scaled < math.MinInt32 { return Q16(math.MinInt32) } "
        "    return Q16(math.Round(scaled)) } "
        "func (q Q16) ToFloat32() float32 { return float32(float64(q) / Q16Scale) } "
        "type Matrix struct { Data []Q16; Rows, Cols int } "
        "func NewMatrix(rows, cols int) *Matrix { "
        "    return &Matrix{Data: make([]Q16, rows*cols), Rows: rows, Cols: cols} } "
        "func (m *Matrix) At(row, col int) Q16 { return m.Data[row*m.Cols+col] } "
        "func (m *Matrix) Set(row, col int, v Q16) { m.Data[row*m.Cols+col] = v } "
        "func Matmul(a, b *Matrix) *Matrix { "
        "    if a.Cols != b.Rows { panic('dimension mismatch') } "
        "    c := NewMatrix(a.Rows, b.Cols) "
        "    var wg sync.WaitGroup "
        "    for i := 0; i < a.Rows; i++ { "
        "        wg.Add(1) "
        "        go func(row int) { "
        "            defer wg.Done() "
        "            for j := 0; j < b.Cols; j++ { "
        "                var acc int64 "
        "                for k := 0; k < a.Cols; k++ { "
        "                    acc += int64(a.At(row, k)) * int64(b.At(k, j)) } "
        "                c.Set(row, j, Q16(acc>>16)) } }(i) } "
        "    wg.Wait() "
        "    return c } "
        "func LayerNorm(x *Matrix, gamma, beta []Q16) *Matrix { "
        "    out := NewMatrix(x.Rows, x.Cols) "
        "    for i := 0; i < x.Rows; i++ { "
        "        var sum int64 "
        "        for j := 0; j < x.Cols; j++ { sum += int64(x.At(i, j)) } "
        "        mean := Q16(sum / int64(x.Cols)) "
        "        var varAcc int64 "
        "        for j := 0; j < x.Cols; j++ { "
        "            d := int64(x.At(i, j)) - int64(mean) "
        "            varAcc += d * d } "
        "        variance := float64(varAcc) / float64(x.Cols) / Q16Scale / Q16Scale "
        "        invStd := 1.0 / math.Sqrt(variance + LayerNormEps) "
        "        invStdQ16 := Float32ToQ16(float32(invStd)) "
        "        for j := 0; j < x.Cols; j++ { "
        "            d := int64(x.At(i, j)) - int64(mean) "
        "            normalized := Q16(d * int64(invStdQ16) >> 16) "
        "            out.Set(i, j, Q16(int64(normalized)*int64(gamma[j])>>16 + int64(beta[j]))) } } "
        "    return out } "
    ),
    # Long multi-topic query
    (
        "Compare and contrast the following approaches to distributed consensus "
        "and explain when to use each one in the context of a globally distributed "
        "database system: Paxos (Classic Paxos, Multi-Paxos, and Fast Paxos variants), "
        "Raft (with and without pre-vote extension), Zab (ZooKeeper Atomic Broadcast), "
        "PBFT (Practical Byzantine Fault Tolerance), HotStuff (and its chained variant "
        "used in LibraBFT/DiemBFT), Tendermint/CometBFT, and Stellar Consensus Protocol. "
        "For each algorithm, discuss: the fault model (crash fault tolerant vs byzantine "
        "fault tolerant), the minimum number of nodes required for fault tolerance "
        "(f+1 vs 2f+1 vs 3f+1), the leader election mechanism and its impact on "
        "availability during leader failures, the message complexity per consensus "
        "round, the latency in terms of network round trips, the throughput in "
        "transactions per second for typical LAN and WAN deployments, the behavior "
        "under network partitions (CAP theorem position), the implementation complexity "
        "and available production-grade open-source implementations, and any known "
        "correctness bugs or safety violations discovered in production deployments. "
        "Then recommend the appropriate algorithm for a financial trading system "
        "requiring sub-millisecond latency with strong consistency guarantees, "
        "a globally distributed key-value store spanning 5 geographic regions "
        "with eventual consistency acceptable, and an open blockchain network "
        "requiring Byzantine fault tolerance with unknown participants."
    ),
    # Another very long Python file
    (
        "from __future__ import annotations "
        "import asyncio "
        "import contextlib "
        "import dataclasses "
        "import hashlib "
        "import json "
        "import logging "
        "import os "
        "import time "
        "from collections.abc import AsyncGenerator, Generator "
        "from pathlib import Path "
        "from typing import Any, ClassVar "
        "import aiohttp "
        "import numpy as np "
        "logger = logging.getLogger(__name__) "
        "@dataclasses.dataclass(frozen=True, slots=True) "
        "class EncoderConfig: "
        "    model_hash: str "
        "    tokenizer_hash: str "
        "    hidden_dim: int = 384 "
        "    num_heads: int = 12 "
        "    num_layers: int = 12 "
        "    max_seq_len: int = 256 "
        "    window_size: int = 256 "
        "    stride: int = 192 "
        "    top_k: int = 5 "
        "    backend: str = 'native' "
        "    runtime_dir: Path = dataclasses.field( "
        "        default_factory=lambda: Path.home() / '.local/share/mind-nerve/runtime') "
        "    SCHEMA_VERSION: ClassVar[int] = 1 "
        "    def to_cache_key(self) -> str: "
        "        payload = json.dumps(dataclasses.asdict(self), sort_keys=True, default=str) "
        "        return hashlib.sha256(payload.encode()).hexdigest() "
        "class EncoderError(Exception): "
        "    pass "
        "class BackendNotAvailableError(EncoderError): "
        "    pass "
        "class EncoderTimeoutError(EncoderError): "
        "    pass "
        "@contextlib.asynccontextmanager "
        "async def managed_encoder(config: EncoderConfig) -> AsyncGenerator[AsyncEncoder, None]: "
        "    encoder = AsyncEncoder(config) "
        "    try: "
        "        await encoder.initialize() "
        "        yield encoder "
        "    finally: "
        "        await encoder.shutdown() "
        "class AsyncEncoder: "
        "    def __init__(self, config: EncoderConfig) -> None: "
        "        self._config = config "
        "        self._session: aiohttp.ClientSession | None = None "
        "        self._initialized = False "
        "        self._request_count = 0 "
        "        self._error_count = 0 "
        "        self._total_latency_ms = 0.0 "
        "    async def initialize(self) -> None: "
        "        connector = aiohttp.TCPConnector(limit=100, limit_per_host=20) "
        "        self._session = aiohttp.ClientSession(connector=connector) "
        "        self._initialized = True "
        "        logger.info('AsyncEncoder initialized: backend=%s', self._config.backend) "
        "    async def encode(self, query: str, timeout: float = 5.0) -> np.ndarray: "
        "        if not self._initialized: "
        "            raise EncoderError('Encoder not initialized') "
        "        start = time.perf_counter() "
        "        try: "
        "            result = await asyncio.wait_for(self._encode_impl(query), timeout=timeout) "
        "            self._total_latency_ms += (time.perf_counter() - start) * 1000 "
        "            self._request_count += 1 "
        "            return result "
        "        except asyncio.TimeoutError as exc: "
        "            self._error_count += 1 "
        "            raise EncoderTimeoutError(f'Encode timed out after {timeout}s') from exc "
        "    async def _encode_impl(self, query: str) -> np.ndarray: "
        "        raise NotImplementedError "
        "    async def shutdown(self) -> None: "
        "        if self._session: "
        "            await self._session.close() "
        "        self._initialized = False "
        "    @property "
        "    def stats(self) -> dict[str, Any]: "
        "        return { 'request_count': self._request_count, "
        "                 'error_count': self._error_count, "
        "                 'error_rate': self._error_count / max(1, self._request_count), "
        "                 'avg_latency_ms': self._total_latency_ms / max(1, self._request_count) } "
    ),
]


# ---------------------------------------------------------------------------
# Adversarial corpus
# ---------------------------------------------------------------------------

def _build_adversarial(rng: _Xorshift64) -> list[str]:
    """200 adversarial cases covering edge conditions in the encoder."""
    cases: list[str] = []

    # 1. Empty string (1 case — [CLS][SEP] path)
    cases.append("")

    # 2. Single-char inputs (20 cases)
    for ch in "abcdefghijklmnopqrst":
        cases.append(ch)

    # 3. Leading/trailing whitespace variants (20 cases)
    base = "git status"
    whitespace_variants = [
        " " + base,
        "  " + base,
        "   " + base,
        base + " ",
        base + "  ",
        " " + base + " ",
        "\t" + base,
        "\n" + base,
        base + "\n",
        "\t" + base + "\t",
        "  " * 5 + base,
        base + "  " * 5,
        "\r\n" + base,
        base + "\r\n",
        " ".join([""] * 10) + base,
        base.upper(),
        base.lower(),
        base.title(),
        base * 2,
        base + " " + base,
    ]
    cases.extend(whitespace_variants)

    # 4. All-punctuation (10 cases)
    punct_cases = [
        "!!!",
        "???",
        "---",
        "...",
        "///",
        "###",
        "@@@",
        "$$$",
        "%%%",
        "^^^",
    ]
    cases.extend(punct_cases)

    # 5. Numeric strings (10 cases)
    num_cases = [
        "0",
        "123",
        "1234567890",
        "3.14159265358979",
        "1e-10",
        "0xDEADBEEF",
        "0b10101010",
        "0o777",
        "1_000_000",
        "inf",
    ]
    cases.extend(num_cases)

    # 6. Unicode edge cases (10 cases)
    unicode_cases = [
        "\x01",        # control character SOH
        "�",          # replacement character
        "​",          # zero-width space
        "‌",          # zero-width non-joiner
        "‍",          # zero-width joiner
        "﻿",          # BOM
        "é",          # é
        "中文",   # Chinese characters
        "ال",   # Arabic
        "αβγ",  # Greek α β γ
    ]
    cases.extend(unicode_cases)

    # 7. Max-length-256 boundary inputs — strings that tokenize to exactly 256
    # We approximate with a 1200-char string (BERT WordPiece ~4.7 chars/token).
    # Exact token count depends on tokenizer; the runner will truncate to max_len.
    max256_base = "analyze " * 150  # ~1200 chars, targets ~256 token window
    cases.append(max256_base.strip())
    max256_code = ("x = " * 64).strip()
    cases.append(max256_code)

    # 8. Max-length-512 boundary inputs — beyond sliding window threshold
    max512_base = "evaluate " * 170  # ~1530 chars, ~340 tokens
    cases.append(max512_base.strip())
    max512_code = ("result = compute(a, b, c) " * 50).strip()
    cases.append(max512_code)

    # 9. All-unknown-token approximation — rare byte sequences that will
    # produce [UNK] tokens when run through BERT WordPiece
    unk_cases = [
        "xyzzy qux zap blorp wibble",
        "aaaaaaaaaaaaaaaaaaaaaaaaa",
        "z" * 50,
        "bbbbbbbbbbbbbbbbbbbbbbbbb",
        "zzzzzzzzzzzzzzzzzzzzzzzzz",
    ]
    cases.extend(unk_cases)

    # 10. Repeated single token (10 cases)
    rep_cases = [
        "the " * 30,
        "is " * 30,
        "a " * 30,
        "of " * 30,
        "and " * 30,
        "to " * 30,
        "in " * 30,
        "that " * 20,
        "it " * 30,
        "with " * 20,
    ]
    cases.extend(s.strip() for s in rep_cases)

    # 11. Code-like strings with special tokens (10 cases)
    code_cases = [
        "def f(): pass",
        "class A(B): ...",
        "import sys; sys.exit(0)",
        "{ 'key': [1, 2, 3] }",
        "SELECT * FROM users WHERE id = 1",
        "curl -X POST http://localhost:8080/api",
        "kubectl get pods --all-namespaces",
        "docker run -it --rm ubuntu:22.04 bash",
        "python3 -c 'print(1+1)'",
        "bash -c 'for i in $(seq 10); do echo $i; done'",
    ]
    cases.extend(code_cases)

    # 12. Pad to exactly 200 cases with synthetic variations
    base_queries = [
        "what is", "how to", "find all", "list", "get", "show me",
        "explain", "help with", "configure", "deploy", "run", "stop",
        "start", "restart", "check", "validate", "test", "build",
        "install", "update",
    ]
    while len(cases) < N_ADVERSARIAL:
        b = base_queries[rng.next_bounded(len(base_queries))]
        suffix = "x" * rng.next_bounded(50)
        q = f"{b} {suffix}"
        cases.append(q)

    return cases[:N_ADVERSARIAL]


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------

def _build_eval_segment(runtime_dir: Path | None, rng: _Xorshift64) -> list[str]:
    """
    Build 600 eval-distribution queries.

    Priority:
    1. Route names from route_table.jsonl (real eval distribution).
    2. Fall back to _HARDCODED_EVAL_SEED if runtime is absent.
    """
    if runtime_dir is not None:
        names = _route_names(runtime_dir)
        if names:
            # Shuffle deterministically, then take first 600.
            rng.shuffle(names)
            names = names[:N_EVAL]
            # Pad if fewer than 600 routes exist.
            seen = set(names)
            names = list(names)
            if len(names) < N_EVAL:
                names.extend(
                    _synthetic_eval_queries(rng, N_EVAL - len(names), seen)
                )
            return names

    # Fallback: hardcoded seed list.
    base = _ensure_600(list(_HARDCODED_EVAL_SEED))
    return base


def _build_long_segment(rng: _Xorshift64) -> list[str]:
    """
    Build 200 long queries (T > 256 tokens).
    Uses the hardcoded templates, shuffled deterministically.
    """
    templates = list(_LONG_QUERY_TEMPLATES)
    rng.shuffle(templates)

    result: list[str] = []
    idx = 0
    while len(result) < N_LONG:
        if idx < len(templates):
            result.append(templates[idx])
            idx += 1
        else:
            # Need more: generate by combining and extending templates.
            base_idx = rng.next_bounded(len(templates))
            base = templates[base_idx]
            # Extend by repeating a tail fragment.
            tail_len = 300
            tail = base[-tail_len:] if len(base) > tail_len else base
            extended = base + " Furthermore, " + tail
            result.append(extended)

    return result[:N_LONG]


def build_corpus() -> list[dict]:
    """
    Build and return the 1,000-query corpus as a list of dicts.
    Each entry: {"id": str, "text": str, "category": str}
    """
    rng = _Xorshift64(_CORPUS_SEED)
    runtime_dir = _resolve_runtime_dir()

    # Segment 1: eval (600)
    eval_segment = _build_eval_segment(runtime_dir, rng)

    # Segment 2: long (200)
    long_segment = _build_long_segment(rng)

    # Segment 3: adversarial (200)
    adv_segment = _build_adversarial(rng)

    entries: list[dict] = []

    for i, text in enumerate(eval_segment):
        entries.append({"id": f"eval_{i:04d}", "text": text, "category": "eval"})

    for i, text in enumerate(long_segment):
        entries.append({"id": f"long_{i:04d}", "text": text, "category": "long"})

    for i, text in enumerate(adv_segment):
        entries.append({"id": f"adv_{i:04d}", "text": text, "category": "adversarial"})

    assert len(entries) == CORPUS_SIZE, f"Expected {CORPUS_SIZE} entries, got {len(entries)}"
    return entries


def write_corpus(path: Path | None = None) -> Path:
    """
    Write the corpus to JSON. Returns the path written.
    Idempotent when called with the same seed.
    """
    out = path or CORPUS_PATH
    entries = build_corpus()
    with out.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    return out


def load_corpus(path: Path | None = None) -> list[dict]:
    """
    Load the corpus from the committed JSON file.
    Falls back to building it on the fly if the file is absent.
    """
    p = path or CORPUS_PATH
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return build_corpus()


def load_long_queries(path: Path | None = None) -> list[dict]:
    """Return only the T > 256 long queries from the corpus."""
    corpus = load_corpus(path)
    return [e for e in corpus if e["category"] == "long"]


if __name__ == "__main__":
    import sys

    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CORPUS_PATH
    written = write_corpus(out_path)
    corpus = load_corpus(written)

    cats: dict[str, int] = {}
    for e in corpus:
        cats[e["category"]] = cats.get(e["category"], 0) + 1

    print(f"Wrote {len(corpus)} queries to {written}")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
