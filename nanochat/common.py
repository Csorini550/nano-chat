"""Common utilities for nanochat."""  # Module-level docstring summarizing shared helpers used in nanochat.

import os  # Access environment variables and filesystem helpers for directory management.
import re  # Provide regular expression support for log message highlighting.
import logging  # Load Python's logging primitives so we can customize formatting and configuration.
import torch  # Import PyTorch to query device capabilities and initialize backends.
import torch.distributed as dist  # Access PyTorch distributed primitives for multi-process coordination.


class ColoredFormatter(logging.Formatter):  # Extend logging.Formatter to add ANSI styling support.
    """Custom formatter that adds colors to log messages."""  # Document that this formatter decorates log output with colors.
    COLORS = {  # Map log level names to ANSI color codes used to colorize log prefixes.
        'DEBUG': '\033[36m',    # Cyan escape sequence for debug messages.
        'INFO': '\033[32m',     # Green escape sequence for informational messages.
        'WARNING': '\033[33m',  # Yellow escape sequence for warnings.
        'ERROR': '\033[31m',    # Red escape sequence for errors.
        'CRITICAL': '\033[35m', # Magenta escape sequence for critical failures.
    }
    RESET = '\033[0m'  # ANSI escape sequence that clears any active styling.
    BOLD = '\033[1m'  # ANSI escape sequence that enables bold text.

    def format(self, record):  # Customize how each LogRecord is rendered into text.
        levelname = record.levelname  # Capture the unmodified severity label so we can both style and inspect it.
        if levelname in self.COLORS:  # Only inject color codes when the level has a configured mapping.
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"  # Wrap the level name in color and bold styling.
        message = super().format(record)  # Defer to the base formatter for the core message content.
        if levelname == 'INFO':  # Further highlight numeric metrics when the message is informational.
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)  # Bold various numeric units and percentages for scanability.
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)  # Color shard identifiers using the info color to keep context clear.
        return message  # Return the fully formatted log string to the logging framework.


def setup_default_logging():  # Build a default logging configuration with our colorized formatter.
    handler = logging.StreamHandler()  # Create a stream handler so logs emit to standard output.
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))  # Attach the colored formatter with a detailed message pattern.
    logging.basicConfig(level=logging.INFO, handlers=[handler])  # Configure the root logger to use INFO level and the single stream handler.


setup_default_logging()  # Initialize the logging system as soon as this module is imported.
logger = logging.getLogger(__name__)  # Acquire a module-level logger for convenience.


def get_base_dir():  # Determine the directory where nanochat should store cache artifacts.
    if os.environ.get("NANOCHAT_BASE_DIR"):  # Prefer a user-specified base directory when provided.
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")  # Read the explicitly configured cache location.
    else:  # Fall back to constructing a cache path under the user's home directory.
        home_dir = os.path.expanduser("~")  # Resolve the absolute path to the current user's home directory.
        cache_dir = os.path.join(home_dir, ".cache")  # Target the conventional ~/.cache folder for intermediate artifacts.
        nanochat_dir = os.path.join(cache_dir, "nanochat")  # Create a nanochat-specific subdirectory inside the cache folder.
    os.makedirs(nanochat_dir, exist_ok=True)  # Ensure the directory exists without raising errors if it already does.
    return nanochat_dir  # Provide callers with the canonical cache directory path.


def print0(s="", **kwargs):  # Convenience wrapper that only prints from the primary distributed rank.
    ddp_rank = int(os.environ.get('RANK', 0))  # Parse the RANK environment variable, defaulting to zero when absent.
    if ddp_rank == 0:  # Restrict printing to rank zero so multi-process logs do not interleave.
        print(s, **kwargs)  # Emit the message using Python's standard print behavior.


def print_banner():  # Emit the nanochat startup banner using the rank-aware printer.
    banner = """
  sSSs   .S    S.    .S_sSSs     .S    sSSs          sSSs   .S    S.    .S_SSSs    sdSS_SSSSSSbs  
 d%%SP  .SS    SS.  .SS~YS%%b   .SS   d%%SP         d%%SP  .SS    SS.  .SS~SSSSS   YSSS~S%SSSSSP  
d%S'    S%S    S%S  S%S   `S%b  S%S  d%S'          d%S'    S%S    S%S  S%S   SSSS       S%S       
S%S     S%S    S%S  S%S    S%S  S%S  S%|           S%S     S%S    S%S  S%S    S%S       S%S       
S&S     S%S SSSS%S  S%S    d*S  S&S  S&S           S&S     S%S SSSS%S  S%S SSSS%S       S&S       
S&S     S&S  SSS&S  S&S   .S*S  S&S  Y&Ss          S&S     S&S  SSS&S  S&S  SSS%S       S&S       
S&S     S&S    S&S  S&S_sdSSS   S&S  `S&&S         S&S     S&S    S&S  S&S    S&S       S&S       
S&S     S&S    S&S  S&S~YSY%b   S&S    `S*S        S&S     S&S    S&S  S&S    S&S       S&S       
S*b     S*S    S*S  S*S   `S%b  S*S     l*S        S*b     S*S    S*S  S*S    S&S       S*S       
S*S.    S*S    S*S  S*S    S%S  S*S    .S*P        S*S.    S*S    S*S  S*S    S*S       S*S       
 SSSbs  S*S    S*S  S*S    S&S  S*S  sSS*S          SSSbs  S*S    S*S  S*S    S*S       S*S       
  YSSP  SSS    S*S  S*S    SSS  S*S  YSS'            YSSP  SSS    S*S  SSS    S*S       S*S       
               SP   SP          SP                                SP          SP        SP        
               Y    Y           Y                                 Y           Y         Y         
                                                                                                  
"""  # Multi-line ASCII art banner that will be displayed at startup.
    print0(banner)  # Print the banner only from rank zero to avoid duplicates.


def is_ddp():  # Check whether the current process is part of a distributed training run.
    return int(os.environ.get('RANK', -1)) != -1  # Report True when the RANK environment variable is set.


def get_dist_info():  # Gather distributed training metadata such as rank and world size.
    if is_ddp():  # When distributed execution is active, read details from the environment.
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])  # Ensure required DDP variables are defined.
        ddp_rank = int(os.environ['RANK'])  # Parse the global rank assigned to this process.
        ddp_local_rank = int(os.environ['LOCAL_RANK'])  # Parse the local rank within the current node.
        ddp_world_size = int(os.environ['WORLD_SIZE'])  # Parse the total number of processes participating.
        return True, ddp_rank, ddp_local_rank, ddp_world_size  # Indicate DDP is active and provide the parsed values.
    else:  # When not running under DDP, provide sane defaults.
        return False, 0, 0, 1  # Signal no DDP and supply placeholder rank and world size values.


def autodetect_device_type():  # Determine the most capable device available on the system.
    if torch.cuda.is_available():  # Prefer CUDA GPUs when the system has them.
        device_type = "cuda"  # Record that CUDA should be used.
    elif torch.backends.mps.is_available():  # Otherwise prefer Apple's Metal Performance Shaders on macOS.
        device_type = "mps"  # Record that MPS should be used.
    else:  # Fall back to CPU execution when no accelerated backends exist.
        device_type = "cpu"  # Record that CPU should be used.
    print0(f"Autodetected device type: {device_type}")  # Inform the user which device was chosen.
    return device_type  # Return the selected device type to the caller.


def compute_init(device_type="cuda"):  # Perform common initialization before running models.
    """Basic initialization that we keep doing over and over, so make common."""  # Explain that this routine centralizes repeated setup work.
    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"  # Guard against unsupported device strings.
    if device_type == "cuda":  # When the user requests CUDA initialization, verify availability.
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"  # Fail fast if CUDA support is missing.
    if device_type == "mps":  # When the user requests MPS initialization, verify availability.
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"  # Fail fast if MPS support is missing.

    torch.manual_seed(42)  # Seed the CPU random number generator for reproducibility.
    if device_type == "cuda":  # Apply GPU seeding only when running on CUDA devices.
        torch.cuda.manual_seed(42)  # Seed the CUDA random number generator to align GPU runs.
    # skipping full reproducibility for now, possibly investigate slowdown later  # Note that deterministic algorithms are intentionally skipped for performance.
    # torch.use_deterministic_algorithms(True)  # Example of how to enforce full determinism if needed in the future.

    if device_type == "cuda":  # Adjust compute precision settings specifically for CUDA.
        torch.set_float32_matmul_precision("high")  # Enable TF32 matmul precision for better performance on Ampere+ GPUs.

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()  # Query the distributed context to see if DDP is active.
    if ddp and device_type == "cuda":  # Only run the CUDA-specific DDP initialization path when required.
        device = torch.device("cuda", ddp_local_rank)  # Select the appropriate CUDA device for this process.
        torch.cuda.set_device(device)  # Make PyTorch default to the selected CUDA device.
        dist.init_process_group(backend="nccl", device_id=device)  # Initialize the NCCL process group for inter-process communication.
        dist.barrier()  # Synchronize all participating ranks before continuing.
    else:  # When not using CUDA-based DDP, fall back to a simpler device selection.
        device = torch.device(device_type)  # Create a torch.device for CPU or MPS execution as requested.

    if ddp_rank == 0:  # Only the primary rank should log the distributed configuration.
        logger.info(f"Distributed world size: {ddp_world_size}")  # Report how many processes are participating in the job.

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device  # Provide the caller with DDP state and the configured device.


def compute_cleanup():  # Reverse the side effects of compute_init when the program exits.
    """Companion function to compute_init, to clean things up before script exit"""  # Document that this routine pairs with compute_init for teardown.
    if is_ddp():  # Only perform cleanup if a distributed process group was initialized.
        dist.destroy_process_group()  # Shut down the NCCL process group to release resources.


class DummyWandb:  # Minimal stub that mimics the wandb module interface.
    """Useful if we wish to not use wandb but have all the same signatures"""  # Explain that this class preserves call signatures without external dependencies.

    def __init__(self):  # Initialize the dummy object without stored state.
        pass  # No state or configuration is necessary.

    def log(self, *args, **kwargs):  # Mirror wandb.log while intentionally ignoring inputs.
        pass  # Ignore logging requests to disable wandb without code changes.

    def finish(self):  # Mirror wandb.finish for API compatibility.
        pass  # Nothing to tear down when wandb is disabled.
