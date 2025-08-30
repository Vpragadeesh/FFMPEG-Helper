#!/usr/bin/env python3
"""
H264 to H265 Video Converter with FZF Integration and Audio Enhancement
Converts video files from H264 codec to H265 (HEVC) with interactive file selection and audio enhancement
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time
import re
from typing import Optional, List, Set, Dict
import json


class VideoConverter:
    def __init__(self, crf: int = 28, preset: str = "medium", keep_original: bool = True,
                 use_gpu: bool = True, gpu_preset: str = "p4", enhance_audio: bool = True,
                 audio_codec: str = "aac", audio_bitrate: str = "128k", normalize_audio: bool = False,
                 denoise_audio: bool = False, audio_quality: str = "medium", debug: bool = False,
                 volume_boost: float = 1.5, enhance_music: bool = False):
        """
        Initialize the video converter.

        Args:
            crf: Constant Rate Factor (0-51, lower = better quality, larger file)
            preset: Encoding preset for CPU (ultrafast, superfast, veryfast, faster, fast,
                   medium, slow, slower, veryslow)
            keep_original: Whether to keep the original file after conversion
            use_gpu: Whether to use NVIDIA GPU hardware acceleration (NVENC)
            gpu_preset: GPU preset (p1-p7, where p1=fastest, p7=slowest/best quality)
            enhance_audio: Whether to enhance audio instead of just copying
            audio_codec: Audio codec to use (aac, mp3, opus, ac3)
            audio_bitrate: Audio bitrate (64k, 128k, 192k, 256k, 320k)
            normalize_audio: Whether to normalize audio levels
            denoise_audio: Whether to apply noise reduction
            audio_quality: Audio quality preset (low, medium, high)
            debug: Enable debug mode for troubleshooting
            volume_boost: Overall volume multiplier (1.0 = no change, 1.5 = 50% louder)
            enhance_music: Whether to enhance background music/frequencies
        """
        self.crf = crf
        self.preset = preset
        self.keep_original = keep_original
        self.use_gpu = use_gpu
        self.gpu_preset = gpu_preset
        self.enhance_audio = enhance_audio
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate
        self.normalize_audio = normalize_audio
        self.denoise_audio = denoise_audio
        self.audio_quality = audio_quality
        self.debug = debug
        self.volume_boost = volume_boost
        self.enhance_music = enhance_music
        self.video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.m4v', '.flv', '.wmv', '.webm'}
        self.gpu_available = False

        # Audio codec settings - improved compatibility
        self.audio_codecs = {
            'aac': 'aac',
            'mp3': 'libmp3lame',
            'opus': 'libopus',
            'ac3': 'ac3',
            'flac': 'flac'
        }

        # Audio quality presets
        self.audio_quality_presets = {
            'low': {'bitrate': '96k', 'quality': '5'},
            'medium': {'bitrate': '128k', 'quality': '3'},
            'high': {'bitrate': '192k', 'quality': '1'}
        }

        # Codec compatibility matrix
        self.codec_filter_compatibility = {
            'flac': {
                'supports_normalization': False,  # FLAC normalization can cause issues
                'supports_denoise': False,       # FLAC denoise often fails
                'requires_special_handling': True,
                'notes': 'Lossless codec - audio filters not recommended. Use for archival quality without processing.'
            },
            'aac': {
                'supports_normalization': True,
                'supports_denoise': True,
                'requires_special_handling': False,
                'notes': 'Best compatibility with all filters'
            },
            'mp3': {
                'supports_normalization': True,
                'supports_denoise': True,
                'requires_special_handling': False,
                'notes': 'Good compatibility, widely supported'
            },
            'opus': {
                'supports_normalization': True,
                'supports_denoise': True,
                'requires_special_handling': False,
                'notes': 'Modern codec with excellent quality'
            },
            'ac3': {
                'supports_normalization': True,
                'supports_denoise': False,
                'requires_special_handling': False,
                'notes': 'Limited filter support'
            }
        }

        self.check_dependencies()

    def check_dependencies(self):
        """Check if FFmpeg, fzf, and GPU encoders are available."""
        # Check FFmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: FFmpeg is not installed or not in PATH")
            print("Please install FFmpeg: https://ffmpeg.org/download.html")
            sys.exit(1)

        # Check fzf
        try:
            subprocess.run(["fzf", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: fzf is not installed or not in PATH")
            print("Please install fzf:")
            print("  Ubuntu/Debian: sudo apt install fzf")
            print("  macOS: brew install fzf")
            print("  Windows: winget install fzf")
            sys.exit(1)

        # Check for NVIDIA GPU encoders
        if self.use_gpu:
            self.gpu_available = self.check_gpu_encoders()
            if self.gpu_available:
                print("🚀 NVIDIA GPU acceleration available!")
            else:
                print("⚠️  GPU acceleration not available, falling back to CPU encoding")
                self.use_gpu = False

    def check_gpu_encoders(self) -> bool:
        """Check if NVIDIA GPU encoders are available."""
        try:
            # Check for NVENC HEVC support
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, check=True
            )

            # Look for hevc_nvenc encoder
            if "hevc_nvenc" in result.stdout:
                # Test if we can actually use it
                test_cmd = [
                    "ffmpeg", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=1",
                    "-c:v", "hevc_nvenc", "-f", "null", "-"
                ]
                test_result = subprocess.run(test_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return test_result.returncode == 0

            return False
        except subprocess.CalledProcessError:
            return False

    def validate_audio_settings(self) -> tuple[bool, str]:
        """Validate current audio settings for compatibility."""
        if not self.enhance_audio:
            return True, "Audio copying enabled - no validation needed"

        codec_info = self.codec_filter_compatibility.get(self.audio_codec, {})
        warnings = []

        # Check normalization compatibility
        if self.normalize_audio and not codec_info.get('supports_normalization', True):
            warnings.append(f"⚠️  {self.audio_codec.upper()} may not work well with normalization")

        # Check denoise compatibility
        if self.denoise_audio and not codec_info.get('supports_denoise', True):
            return False, f"❌ {self.audio_codec.upper()} does not support noise reduction filters"

        # Special handling warnings
        if codec_info.get('requires_special_handling', False):
            warnings.append(f"ℹ️  {codec_info.get('notes', '')}")

        if warnings:
            return True, "\n".join(warnings)

        return True, "✅ Audio settings are compatible"

    def find_video_files(self, directory: str = ".", recursive: bool = True) -> List[str]:
        """Find all video files in the given directory."""
        video_files = []
        search_path = Path(directory)

        if recursive:
            # Recursively search for video files
            for ext in self.video_extensions:
                video_files.extend(search_path.rglob(f"*{ext}"))
        else:
            # Search only in the current directory
            for ext in self.video_extensions:
                video_files.extend(search_path.glob(f"*{ext}"))

        # Convert to strings and sort
        return sorted([str(f) for f in video_files])

    def is_h264_video(self, video_file: str) -> bool:
        """Check if a video file uses H264 codec."""
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name", "-of", "csv=p=0",
            video_file
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            codec = result.stdout.strip().lower()
            return codec == "h264"
        except subprocess.CalledProcessError:
            return False

    def get_video_info(self, video_file: str) -> dict:
        """Get detailed information about a video file."""
        cmd = [
            "ffprobe", "-v", "error", "-show_format", "-show_streams",
            "-of", "json", video_file
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            # Extract relevant info
            format_info = data.get("format", {})
            video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
            audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})

            return {
                "codec": video_stream.get("codec_name", "unknown"),
                "duration": float(format_info.get("duration", 0)),
                "size": int(format_info.get("size", 0)),
                "width": video_stream.get("width", 0),
                "height": video_stream.get("height", 0),
                "bit_rate": int(format_info.get("bit_rate", 0)) if format_info.get("bit_rate") else 0,
                "audio_codec": audio_stream.get("codec_name", "unknown"),
                "audio_bitrate": int(audio_stream.get("bit_rate", 0)) if audio_stream.get("bit_rate") else 0,
                "audio_channels": audio_stream.get("channels", 0),
                "audio_sample_rate": audio_stream.get("sample_rate", "unknown")
            }
        except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError):
            return {}

    def format_file_for_fzf(self, video_file: str) -> str:
        """Format video file information for fzf display."""
        path = Path(video_file)
        info = self.get_video_info(video_file)

        # Format file size
        size_str = self.format_size(info.get("size", 0))

        # Format duration
        duration = info.get("duration", 0)
        duration_str = f"{int(duration//60):02d}:{int(duration%60):02d}" if duration > 0 else "??:??"

        # Format resolution
        width = info.get("width", 0)
        height = info.get("height", 0)
        resolution = f"{width}x{height}" if width and height else "????x????"

        # Format video codec
        video_codec = info.get("codec", "unknown").upper()

        # Format audio codec
        audio_codec = info.get("audio_codec", "unknown").upper()
        audio_channels = info.get("audio_channels", 0)
        audio_info = f"{audio_codec}" + (f"/{audio_channels}ch" if audio_channels > 0 else "")

        # Create formatted string
        return f"{path.name:<35} | {video_codec:<6} | {audio_info:<10} | {resolution:<9} | {duration_str:<6} | {size_str:<8} | {video_file}"

    def build_audio_filters(self) -> tuple[List[str], bool]:
        """Build FFmpeg audio filter chain based on settings."""
        if not self.enhance_audio:
            return [], True

        audio_filters = []
        success = True

        # Check codec compatibility first
        codec_info = self.codec_filter_compatibility.get(self.audio_codec, {})

        # Special handling for FLAC - automatically disable filters
        if self.audio_codec == 'flac' and (self.denoise_audio or self.normalize_audio):
            print(f"ℹ️  FLAC codec detected - audio filters automatically disabled")
            print(f"   FLAC is lossless and doesn't work well with audio processing")
            return [], True  # No filters for FLAC

        # Add denoise filter first (if supported and requested)
        if self.denoise_audio:
            if codec_info.get('supports_denoise', True):
                # Use a more conservative denoise setting for better compatibility
                audio_filters.append("anlmdn=s=0.001:p=0.90:r=0.001:m=10")
            else:
                print(f"⚠️  Skipping noise reduction - not compatible with {self.audio_codec.upper()}")

        # Add music enhancement filter (if requested)
        if self.enhance_music:
            # Enhance mid-range frequencies where music typically sits (200Hz-8kHz)
            # Boost these frequencies by 3dB while slightly reducing very low and very high frequencies
            music_eq = "equalizer=f=1000:width_type=h:width=4000:g=3,equalizer=f=4000:width_type=h:width=2000:g=2"
            audio_filters.append(music_eq)

        # Add volume boost (if not 1.0)
        if self.volume_boost != 1.0:
            # Use volume filter to boost overall audio
            audio_filters.append(f"volume={self.volume_boost}")

        # Add normalization filter (if supported and requested)
        if self.normalize_audio:
            if codec_info.get('supports_normalization', True):
                # Standard normalization for lossy codecs
                audio_filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
            else:
                print(f"⚠️  Skipping normalization - not compatible with {self.audio_codec.upper()}")

        return audio_filters, success

    def get_audio_codec_args(self) -> List[str]:
        """Get audio codec arguments based on settings."""
        if not self.enhance_audio:
            return ["-c:a", "copy"]

        ffmpeg_codec = self.audio_codecs.get(self.audio_codec, "aac")
        args = ["-c:a", ffmpeg_codec]

        # Handle FLAC specially (lossless)
        if self.audio_codec == "flac":
            args.extend(["-compression_level", "5"])  # Balanced compression
            return args

        # Add bitrate for lossy codecs
        if self.audio_quality in self.audio_quality_presets:
            bitrate = self.audio_quality_presets[self.audio_quality]["bitrate"]
        else:
            bitrate = self.audio_bitrate
        args.extend(["-b:a", bitrate])

        # Add quality settings for specific codecs
        if self.audio_codec == "aac":
            if self.audio_quality in self.audio_quality_presets:
                quality = self.audio_quality_presets[self.audio_quality]["quality"]
                args.extend(["-q:a", quality])
        elif self.audio_codec == "mp3":
            args.extend(["-q:a", "2"])  # High quality MP3
        elif self.audio_codec == "opus":
            args.extend(["-vbr", "on", "-compression_level", "10"])

        return args

    def select_files_with_fzf(self, directory: str = ".", recursive: bool = True,
                             h264_only: bool = False) -> List[str]:
        """Use fzf to interactively select video files."""
        print("🔍 Scanning for video files...")
        video_files = self.find_video_files(directory, recursive)

        if not video_files:
            print("No video files found!")
            return []

        # Filter H264 files if requested
        if h264_only:
            print("🎯 Filtering H264 videos...")
            h264_files = []
            for file in video_files:
                if self.is_h264_video(file):
                    h264_files.append(file)
            video_files = h264_files

            if not video_files:
                print("No H264 video files found!")
                return []

        print(f"Found {len(video_files)} video file(s)")
        print("📋 Preparing file list for selection...")

        # Format files for fzf display
        formatted_files = []
        for video_file in video_files:
            formatted_files.append(self.format_file_for_fzf(video_file))

        # Create fzf input
        fzf_input = "\n".join(formatted_files)

        # Run fzf
        fzf_cmd = [
            "fzf",
            "--multi",  # Allow multiple selections
            "--preview-window", "up:3:wrap",
            "--preview", "echo 'File: {}' && ffprobe -v error -show_format -show_streams '{}' 2>/dev/null | grep -E '(codec_name|duration|width|height|bit_rate|channels)' | head -15",
            "--header", "Select video files to convert (Space/Tab to select multiple, Enter to confirm)",
            "--prompt", "Videos> ",
            "--height", "40%"
        ]

        try:
            process = subprocess.Popen(
                fzf_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            stdout, stderr = process.communicate(input=fzf_input)

            if process.returncode == 0 and stdout.strip():
                # Extract actual file paths from fzf output
                selected_files = []
                for line in stdout.strip().split('\n'):
                    # Extract the file path (last part after the last |)
                    file_path = line.split(' | ')[-1].strip()
                    selected_files.append(file_path)
                return selected_files
            else:
                print("No files selected or fzf cancelled")
                return []

        except Exception as e:
            print(f"Error running fzf: {e}")
            return []

    def get_video_duration(self, input_file: str) -> Optional[float]:
        """Get the duration of a video file in seconds."""
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            input_file
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            return None

    def parse_progress(self, line: str, duration: float) -> Optional[float]:
        """Parse FFmpeg output to get conversion progress."""
        time_match = re.search(r"time=(\d{2}):(\d{2}):(\d{2}\.\d{2})", line)
        if time_match and duration > 0:
            hours, minutes, seconds = time_match.groups()
            current_time = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            return (current_time / duration) * 100
        return None

    def convert_file(self, input_file: str, output_file: Optional[str] = None) -> bool:
        """
        Convert a single video file from H264 to H265.

        Args:
            input_file: Path to the input video file
            output_file: Path to the output file (optional)

        Returns:
            True if conversion successful, False otherwise
        """
        input_path = Path(input_file)

        if not input_path.exists():
            print(f"Error: Input file '{input_file}' does not exist")
            return False

        # Validate audio settings before conversion
        valid, message = self.validate_audio_settings()
        if not valid:
            print(f"❌ Audio settings validation failed: {message}")
            return False
        elif self.enhance_audio and message != "✅ Audio settings are compatible":
            print(f"⚠️  Audio settings warning: {message}")

        # Generate output filename if not provided
        if output_file is None:
            suffix = "-h265-enhanced" if self.enhance_audio else "_h265"
            output_file = str(input_path.with_stem(f"{input_path.stem}{suffix}"))

        # Get video duration for progress tracking
        duration = self.get_video_duration(input_file)

        # Build audio filter chain
        audio_filters, filter_success = self.build_audio_filters()
        if not filter_success:
            print("❌ Failed to build audio filter chain")
            return False

        audio_codec_args = self.get_audio_codec_args()

        # Build FFmpeg command with GPU or CPU encoding
        if self.use_gpu and self.gpu_available:
            # NVIDIA GPU encoding with NVENC
            cmd = [
                "ffmpeg",
                "-hwaccel", "cuda",  # Hardware acceleration
                "-i", input_file,
                "-c:v", "hevc_nvenc",  # Use NVIDIA HEVC encoder
                "-preset", self.gpu_preset,  # GPU preset (p1-p7)
                "-cq", str(self.crf),  # Quality setting for NVENC (use -cq instead of -crf)
            ]
            encoder_info = f"NVENC GPU (Preset: {self.gpu_preset})"
        else:
            # CPU encoding (fallback)
            cmd = [
                "ffmpeg",
                "-i", input_file,
                "-c:v", "libx265",  # Use CPU H265 codec
                "-crf", str(self.crf),  # Quality setting
                "-preset", self.preset,  # Encoding speed preset
            ]
            encoder_info = f"CPU (Preset: {self.preset})"

        # Add audio processing
        if audio_filters and self.enhance_audio:
            cmd.extend(["-af", ",".join(audio_filters)])

        cmd.extend(audio_codec_args)
        cmd.extend([
            "-tag:v", "hvc1",  # Tag for better compatibility
            "-movflags", "+faststart",  # Optimize for streaming
            "-y",  # Overwrite output file if exists
            output_file
        ])

        # Display conversion info
        print(f"\n🎬 Converting: {input_path.name}")
        print(f"📁 Output: {Path(output_file).name}")
        print(f"⚙️  Video: CRF={self.crf}, {encoder_info}")

        if self.enhance_audio:
            audio_info = f"🎵 Audio: {self.audio_codec.upper()}"
            if self.audio_codec != "flac":
                audio_info += f" @ {self.audio_bitrate}"
            if self.volume_boost != 1.0:
                audio_info += f" + Volume Boost ({self.volume_boost}x)"
            if self.enhance_music:
                audio_info += " + Music Enhancement"
            if self.normalize_audio:
                audio_info += " + Normalization"
            if self.denoise_audio:
                audio_info += " + Noise Reduction"
            print(f"   {audio_info}")
        else:
            print("   🎵 Audio: Copy (no enhancement)")

        if self.debug:
            print(f"\n🐛 Debug: FFmpeg command:")
            print(f"   {' '.join(cmd)}")

        print("-" * 60)

        try:
            # Run FFmpeg with progress tracking
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # Monitor progress and collect stderr for error reporting
            start_time = time.time()
            stderr_output = []

            while True:
                line = process.stderr.readline()
                if not line:
                    break

                stderr_output.append(line)

                # Parse and display progress
                if duration:
                    progress = self.parse_progress(line, duration)
                    if progress:
                        bar_length = 50
                        filled = int(bar_length * progress / 100)
                        bar = "█" * filled + "░" * (bar_length - filled)
                        elapsed = time.time() - start_time
                        eta = (elapsed / progress * 100 - elapsed) if progress > 0 else 0
                        print(f"\r🔄 [{bar}] {progress:.1f}% | ETA: {int(eta//60):02d}:{int(eta%60):02d}", end="", flush=True)

            # Wait for process to complete
            process.wait()

            if process.returncode == 0:
                print(f"\n✅ Successfully converted: {Path(output_file).name}")

                # Show file size comparison
                original_size = input_path.stat().st_size
                new_size = Path(output_file).stat().st_size
                change = (1 - new_size / original_size) * 100

                print(f"   📊 Original size: {self.format_size(original_size)}")
                print(f"   📊 New size: {self.format_size(new_size)}")
                print(f"   📉 Size change: {change:+.1f}%")

                # Delete original if requested
                if not self.keep_original and input_file != output_file:
                    os.remove(input_file)
                    print(f"   🗑️  Original file deleted")

                return True
            else:
                print(f"\n❌ Error converting {input_file}")
                print(f"   Return code: {process.returncode}")

                # Analyze stderr for common issues
                stderr_text = "".join(stderr_output)
                if self.debug:
                    print(f"\n🐛 Debug: FFmpeg stderr output:")
                    print(stderr_text)

                # Provide helpful error messages
                if "Invalid data found when processing input" in stderr_text:
                    print("   💡 Suggestion: The input file may be corrupted or unsupported")
                elif "Codec not currently supported" in stderr_text:
                    print("   💡 Suggestion: Try a different audio codec (aac, mp3)")
                elif "filter" in stderr_text.lower() and "error" in stderr_text.lower():
                    print("   💡 Suggestion: Try disabling audio filters (normalization/denoise)")
                elif "Permission denied" in stderr_text:
                    print("   💡 Suggestion: Check file permissions and disk space")
                else:
                    print("   💡 Suggestion: Try with simpler settings or enable debug mode (--debug)")

                return False

        except Exception as e:
            print(f"\n❌ Error during conversion: {e}")
            return False

    def convert_batch(self, input_files: List[str], output_dir: Optional[str] = None) -> None:
        """
        Convert multiple video files.

        Args:
            input_files: List of input file paths
            output_dir: Output directory (optional)
        """
        successful = 0
        failed = 0
        total_original_size = 0
        total_new_size = 0

        print(f"\n🚀 Batch conversion: {len(input_files)} file(s)")
        print("=" * 60)

        start_time = time.time()

        for i, input_file in enumerate(input_files, 1):
            print(f"\n📹 File {i}/{len(input_files)}")

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                suffix = "-h265-enhanced" if self.enhance_audio else "_h265"
                output_file = os.path.join(
                    output_dir,
                    Path(input_file).stem + suffix + Path(input_file).suffix
                )
            else:
                output_file = None

            # Track file sizes for batch statistics
            original_size = Path(input_file).stat().st_size if Path(input_file).exists() else 0
            total_original_size += original_size

            if self.convert_file(input_file, output_file):
                successful += 1
                if output_file and Path(output_file).exists():
                    total_new_size += Path(output_file).stat().st_size
                else:
                    # If output_file is None, calculate the expected output filename
                    suffix = "-h265-enhanced" if self.enhance_audio else "_h265"
                    expected_output = str(Path(input_file).with_stem(f"{Path(input_file).stem}{suffix}"))
                    if Path(expected_output).exists():
                        total_new_size += Path(expected_output).stat().st_size
            else:
                failed += 1

        elapsed_time = time.time() - start_time
        total_change = (1 - total_new_size / total_original_size) * 100 if total_original_size > 0 else 0

        print("\n" + "=" * 60)
        print(f"🎉 Batch conversion complete!")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏱️  Total time: {int(elapsed_time//60):02d}:{int(elapsed_time%60):02d}")
        print(f"   📊 Total original size: {self.format_size(total_original_size)}")
        print(f"   📊 Total new size: {self.format_size(total_new_size)}")
        print(f"   📉 Total size change: {total_change:+.1f}%")

    def interactive_mode(self, directory: str = ".", recursive: bool = True, h264_only: bool = False):
        """Interactive mode using fzf for file selection."""
        print("🎯 Interactive Video Converter with Audio Enhancement")
        print("=" * 50)

        while True:
            # Select files with fzf
            selected_files = self.select_files_with_fzf(directory, recursive, h264_only)

            if not selected_files:
                print("👋 Goodbye!")
                break

            print(f"\n📋 Selected {len(selected_files)} file(s):")
            for i, file in enumerate(selected_files, 1):
                print(f"   {i}. {Path(file).name}")

            # Ask for output directory
            output_dir = input("\n📁 Output directory (Enter for same directory): ").strip()
            if not output_dir:
                output_dir = None
            elif output_dir and not os.path.exists(output_dir):
                create = input(f"Directory '{output_dir}' doesn't exist. Create it? (y/N): ").strip().lower()
                if create in ['y', 'yes']:
                    os.makedirs(output_dir, exist_ok=True)
                else:
                    output_dir = None

            # Ask for settings confirmation
            print(f"\n⚙️  Current settings:")
            print(f"   Video Encoder: {'🚀 GPU (NVENC)' if self.use_gpu and self.gpu_available else '🖥️  CPU'}")
            print(f"   Video CRF: {self.crf} (quality)")
            if self.use_gpu and self.gpu_available:
                print(f"   GPU Preset: {self.gpu_preset} (speed)")
            else:
                print(f"   CPU Preset: {self.preset} (speed)")
            print(f"   Keep original: {self.keep_original}")
            print(f"   🎵 Audio Enhancement: {'✅ Enabled' if self.enhance_audio else '❌ Disabled'}")
            if self.enhance_audio:
                print(f"      Codec: {self.audio_codec.upper()}")
                if self.audio_codec != "flac":
                    print(f"      Bitrate: {self.audio_bitrate}")
                print(f"      Quality: {self.audio_quality}")
                print(f"      Volume Boost: {self.volume_boost}x")
                print(f"      Music Enhancement: {'✅' if self.enhance_music else '❌'}")
                print(f"      Normalize: {'✅' if self.normalize_audio else '❌'}")
                print(f"      Denoise: {'✅' if self.denoise_audio else '❌'}")

                # Show audio settings validation
                valid, message = self.validate_audio_settings()
                if not valid:
                    print(f"      ❌ {message}")
                elif "⚠️" in message or "ℹ️" in message:
                    print(f"      {message}")

            change_settings = input("\nChange settings? (y/N): ").strip().lower()
            if change_settings in ['y', 'yes']:
                self.configure_settings()

            # Confirm conversion
            confirm = input(f"\n🚀 Start conversion of {len(selected_files)} file(s)? (Y/n): ").strip().lower()
            if confirm in ['', 'y', 'yes']:
                if len(selected_files) == 1:
                    output_file = None
                    if output_dir:
                        suffix = "-h265-enhanced" if self.enhance_audio else "_h265"
                        output_file = os.path.join(
                            output_dir,
                            Path(selected_files[0]).stem + suffix + Path(selected_files[0]).suffix
                        )
                    self.convert_file(selected_files[0], output_file)
                else:
                    self.convert_batch(selected_files, output_dir)

            # Ask if user wants to continue
            continue_conv = input(f"\n🔄 Convert more files? (y/N): ").strip().lower()
            if continue_conv not in ['y', 'yes']:
                print("👋 Goodbye!")
                break

    def configure_settings(self):
        """Interactive settings configuration."""
        print("\n⚙️  Configure Settings")
        print("-" * 30)

        # GPU/CPU selection
        if self.gpu_available:
            gpu_input = input(f"Use GPU acceleration? (current: {self.use_gpu}) [Y/n]: ").strip().lower()
            if gpu_input in ['n', 'no', 'false']:
                self.use_gpu = False
            elif gpu_input in ['y', 'yes', 'true', '']:
                self.use_gpu = True

        # CRF setting
        while True:
            try:
                crf_input = input(f"CRF (0-51, current: {self.crf}): ").strip()
                if not crf_input:
                    break
                new_crf = int(crf_input)
                if 0 <= new_crf <= 51:
                    self.crf = new_crf
                    break
                else:
                    print("CRF must be between 0 and 51")
            except ValueError:
                print("Please enter a valid number")

        # Preset setting (GPU or CPU)
        if self.use_gpu and self.gpu_available:
            gpu_presets = ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]
            print(f"\nGPU presets: {', '.join(gpu_presets)} (p1=fastest, p7=best quality)")
            preset_input = input(f"GPU Preset (current: {self.gpu_preset}): ").strip()
            if preset_input and preset_input in gpu_presets:
                self.gpu_preset = preset_input
        else:
            cpu_presets = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"]
            print(f"\nCPU presets: {', '.join(cpu_presets)}")
            preset_input = input(f"CPU Preset (current: {self.preset}): ").strip()
            if preset_input and preset_input in cpu_presets:
                self.preset = preset_input

        # Keep original setting
        keep_input = input(f"Keep original files? (current: {self.keep_original}) [Y/n]: ").strip().lower()
        if keep_input in ['n', 'no', 'false']:
            self.keep_original = False
        elif keep_input in ['y', 'yes', 'true', '']:
            self.keep_original = True

        # Audio enhancement settings
        print(f"\n🎵 Audio Enhancement Settings")
        enhance_input = input(f"Enable audio enhancement? (current: {self.enhance_audio}) [Y/n]: ").strip().lower()
        if enhance_input in ['n', 'no', 'false']:
            self.enhance_audio = False
        elif enhance_input in ['y', 'yes', 'true', '']:
            self.enhance_audio = True

        if self.enhance_audio:
            codecs = list(self.audio_codecs.keys())
            print(f"\nAvailable audio codecs: {', '.join(codecs)}")
            print("  📋 Recommendations:")
            print("     🔸 aac: Best compatibility, good quality, supports all filters")
            print("     🔸 opus: Modern, excellent quality, supports all filters")
            print("     🔸 mp3: Universal support, good quality, supports all filters")
            print("     🔸 flac: Lossless quality, NO FILTERS (archival use)")
            print("     🔸 ac3: Limited filter support")
            print("  ⚠️  Note: FLAC doesn't work with audio filters (normalization/denoise)")

            codec_input = input(f"Audio codec (current: {self.audio_codec}): ").strip().lower()
            if codec_input and codec_input in self.audio_codecs:
                self.audio_codec = codec_input

                # Warn about FLAC limitations
                if self.audio_codec == 'flac':
                    print("  ⚠️  FLAC selected: Audio filters (normalization/denoise) will be disabled")
                    print("     FLAC is for archival quality without processing")

            # Audio quality preset
            if self.audio_codec != "flac":
                qualities = list(self.audio_quality_presets.keys())
                print(f"\nAudio quality presets: {', '.join(qualities)}")
                quality_input = input(f"Audio quality (current: {self.audio_quality}): ").strip().lower()
                if quality_input and quality_input in self.audio_quality_presets:
                    self.audio_quality = quality_input
                    # Update bitrate based on quality preset
                    self.audio_bitrate = self.audio_quality_presets[quality_input]["bitrate"]
                else:
                    # Custom bitrate
                    bitrate_input = input(f"Custom audio bitrate (current: {self.audio_bitrate}): ").strip()
                    if bitrate_input:
                        self.audio_bitrate = bitrate_input

            # Volume boost setting
            print(f"\n🔊 Volume Enhancement")
            while True:
                try:
                    volume_input = input(f"Volume boost multiplier (current: {self.volume_boost}x): ").strip()
                    if not volume_input:
                        break
                    new_volume = float(volume_input)
                    if 0.1 <= new_volume <= 5.0:
                        self.volume_boost = new_volume
                        break
                    else:
                        print("Volume boost must be between 0.1 and 5.0")
                except ValueError:
                    print("Please enter a valid number")

            # Music enhancement setting
            music_input = input(f"Enhance background music? (current: {self.enhance_music}) [y/N]: ").strip().lower()
            self.enhance_music = music_input in ['y', 'yes', 'true']

            # Audio normalization
            if self.audio_codec != 'flac':
                norm_input = input(f"Normalize audio? (current: {self.normalize_audio}) [y/N]: ").strip().lower()
                self.normalize_audio = norm_input in ['y', 'yes', 'true']
            else:
                self.normalize_audio = False
                print("  ℹ️  Audio normalization disabled for FLAC")

            # Audio denoising with compatibility check
            codec_info = self.codec_filter_compatibility.get(self.audio_codec, {})
            if codec_info.get('supports_denoise', True):
                denoise_input = input(f"Apply noise reduction? (current: {self.denoise_audio}) [y/N]: ").strip().lower()
                self.denoise_audio = denoise_input in ['y', 'yes', 'true']
            else:
                print(f"  ℹ️  Noise reduction disabled for {self.audio_codec.upper()}")
                self.denoise_audio = False

        # Validate settings after configuration
        if self.enhance_audio:
            valid, message = self.validate_audio_settings()
            if not valid:
                print(f"\n❌ Configuration error: {message}")
                print("Please reconfigure audio settings.")
                return self.configure_settings()
            elif "⚠️" in message or "ℹ️" in message:
                print(f"\n{message}")

        print(f"\n✅ Settings updated:")
        print(f"   Video Encoder: {'GPU (NVENC)' if self.use_gpu and self.gpu_available else 'CPU'}")
        print(f"   Video CRF: {self.crf}")
        if self.use_gpu and self.gpu_available:
            print(f"   GPU Preset: {self.gpu_preset}")
        else:
            print(f"   CPU Preset: {self.preset}")
        print(f"   Keep original: {self.keep_original}")
        print(f"   🎵 Audio Enhancement: {'Enabled' if self.enhance_audio else 'Disabled'}")
        if self.enhance_audio:
            print(f"      Codec: {self.audio_codec.upper()}")
            if self.audio_codec != "flac":
                print(f"      Bitrate: {self.audio_bitrate}")
            print(f"      Quality: {self.audio_quality}")
            print(f"      Volume Boost: {self.volume_boost}x")
            print(f"      Music Enhancement: {self.enhance_music}")
            print(f"      Normalize: {self.normalize_audio}")
            print(f"      Denoise: {self.denoise_audio}")

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"


def main():
    parser = argparse.ArgumentParser(
        description="Convert H264 videos to H265 (HEVC) format with FZF integration and audio enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Interactive mode with fzf (audio enhancement enabled)
  %(prog)s -i                           # Interactive mode
  %(prog)s video.mp4                    # Convert single file with audio enhancement
  %(prog)s video1.mp4 video2.mkv        # Convert multiple files with audio enhancement
  %(prog)s --no-enhance-audio video.mp4 # Convert without audio enhancement (copy audio)
  %(prog)s -r /videos                   # Interactive mode, search /videos recursively
  %(prog)s --h264-only -i               # Interactive mode, show only H264 files
  %(prog)s -c 23 -p slow video.mp4      # High quality, slow encoding
  %(prog)s -o /output/dir video.mp4     # Specify output directory
  %(prog)s --delete video.mp4           # Delete original after conversion
  %(prog)s --audio-codec aac --audio-bitrate 192k --normalize-audio video.mp4
  %(prog)s --volume-boost 2.0 --enhance-music video.mp4    # Extra volume boost and music enhancement
  %(prog)s --debug video.mp4            # Enable debug mode for troubleshooting
        """
    )

    parser.add_argument(
        "input_files",
        nargs="*",
        help="Input video file(s) to convert (if none provided, use interactive mode)"
    )

    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Force interactive mode with fzf file selection"
    )

    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        default=True,
        help="Search recursively for video files (default: True)"
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search recursively (current directory only)"
    )

    parser.add_argument(
        "--h264-only",
        action="store_true",
        help="Show only H264 videos in interactive mode"
    )

    parser.add_argument(
        "-d", "--directory",
        default=".",
        help="Directory to search for videos (default: current directory)"
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (use CPU encoding)"
    )

    parser.add_argument(
        "--gpu-preset",
        default="p4",
        choices=["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
        help="GPU encoding preset (p1=fastest, p7=best quality, default: p4)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output directory or file (for single input)"
    )

    parser.add_argument(
        "-c", "--crf",
        type=int,
        default=28,
        choices=range(0, 52),
        metavar="CRF",
        help="Constant Rate Factor (0-51, default: 28). Lower = better quality"
    )

    parser.add_argument(
        "-p", "--preset",
        default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast",
                "medium", "slow", "slower", "veryslow"],
        help="Encoding preset (default: medium). Slower = better compression"
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete original files after successful conversion"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for troubleshooting"
    )

    # Audio enhancement arguments
    parser.add_argument(
        "--enhance-audio",
        action="store_true",
        default=True,
        help="Enable audio enhancement instead of copying (default: enabled)"
    )

    parser.add_argument(
        "--no-enhance-audio",
        action="store_true",
        help="Disable audio enhancement (copy audio without processing)"
    )

    parser.add_argument(
        "--audio-codec",
        default="aac",
        choices=["aac", "mp3", "opus", "ac3", "flac"],
        help="Audio codec to use (default: aac)"
    )

    parser.add_argument(
        "--audio-bitrate",
        default="128k",
        help="Audio bitrate (e.g., 64k, 128k, 192k, 256k, 320k) (default: 128k)"
    )

    parser.add_argument(
        "--audio-quality",
        default="medium",
        choices=["low", "medium", "high"],
        help="Audio quality preset (default: medium)"
    )

    parser.add_argument(
        "--normalize-audio",
        action="store_true",
        help="Normalize audio levels"
    )

    parser.add_argument(
        "--denoise-audio",
        action="store_true",
        help="Apply noise reduction to audio"
    )

    parser.add_argument(
        "--volume-boost",
        type=float,
        default=1.5,
        help="Volume boost multiplier (0.1-5.0, default: 1.5). 1.0 = no change, 1.5 = 50%% louder"
    )

    parser.add_argument(
        "--enhance-music",
        action="store_true",
        help="Enhance background music frequencies"
    )

    args = parser.parse_args()

    # Handle recursive flag
    recursive = args.recursive and not args.no_recursive

    # Handle audio enhancement flag (default True, but can be disabled)
    enhance_audio = args.enhance_audio and not args.no_enhance_audio

    # Initialize converter
    converter = VideoConverter(
        crf=args.crf,
        preset=args.preset,
        keep_original=not args.delete,
        use_gpu=not args.no_gpu,
        gpu_preset=args.gpu_preset,
        enhance_audio=enhance_audio,
        audio_codec=args.audio_codec,
        audio_bitrate=args.audio_bitrate,
        normalize_audio=args.normalize_audio,
        denoise_audio=args.denoise_audio,
        audio_quality=args.audio_quality,
        debug=args.debug,
        volume_boost=args.volume_boost,
        enhance_music=args.enhance_music
    )

    # Determine mode
    if not args.input_files or args.interactive:
        # Interactive mode with fzf
        converter.interactive_mode(args.directory, recursive, args.h264_only)
    else:
        # Command line mode (original behavior)
        input_files = []
        for file_pattern in args.input_files:
            path = Path(file_pattern)
            if path.exists():
                input_files.append(str(path))
            else:
                print(f"⚠️  Warning: File not found: {file_pattern}")

        if not input_files:
            print("❌ Error: No valid input files found")
            sys.exit(1)

        # Convert files
        if len(input_files) == 1 and args.output and not os.path.isdir(args.output):
            # Single file with specific output filename
            converter.convert_file(input_files[0], args.output)
        else:
            # Batch conversion
            converter.convert_batch(input_files, args.output)

if __name__ == "__main__":
    main()
