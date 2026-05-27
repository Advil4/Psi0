#!/usr/bin/env python3
"""
远程 PSI0 推理服务启动脚本
在 GPU 服务器上运行，提供 HTTP API 供本地仿真调用
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Start PSI0 Inference Server")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Training run directory (e.g., .runs/finetune/pick-up.real...)"
    )
    parser.add_argument(
        "--ckpt-step",
        type=str,
        default="latest",
        help="Checkpoint step number or 'latest'"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8014,
        help="Server port"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (use 0.0.0.0 for remote access)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device"
    )
    parser.add_argument(
        "--rtc",
        action="store_true",
        help="Enable Real-Time Control (RTC) mode"
    )
    parser.add_argument(
        "--action-exec-horizon",
        type=int,
        default=3,
        help="Number of actions to execute per chunk (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Validate run directory
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"❌ Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        print(f"❌ Error: Checkpoints directory not found: {checkpoints_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("🚀 PSI0 Inference Server")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Checkpoint: {args.ckpt_step}")
    print(f"Host: {args.host}:{args.port}")
    print(f"Device: {args.device}")
    print(f"RTC: {args.rtc}")
    print(f"Action exec horizon: {args.action_exec_horizon}")
    print("=" * 80)
    
    # Import and start server
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    from psi.deploy.psi0_serve_simple import Server
    
    try:
        server = Server(
            policy=None,  # Will be auto-detected from run_dir
            run_dir=run_dir,
            ckpt_step=args.ckpt_step,
            device=args.device,
            enable_rtc=args.rtc,
            action_exec_horizon=args.action_exec_horizon
        )
        
        print("\n✅ Server initialized successfully!")
        print(f"📡 Listening on {args.host}:{args.port}")
        print(f"🔗 API endpoint: http://{args.host}:{args.port}/act")
        print(f"❤️  Health check: http://{args.host}:{args.port}/health")
        print("\nPress Ctrl+C to stop\n")
        
        server.run(host=args.host, port=args.port)
        
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()