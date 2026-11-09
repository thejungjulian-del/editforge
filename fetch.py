"""Download a reference video from a TikTok / Instagram Reels / YouTube Shorts link."""
from pathlib import Path

import yt_dlp

SUPPORTED = ("tiktok.com", "instagram.com", "youtube.com", "youtu.be")


class FetchError(Exception):
    pass


def looks_supported(url):
    return any(host in url.lower() for host in SUPPORTED)


def fetch(url, dest):
    """Download the video at `url` to exactly `dest` (an .mp4 path). Returns metadata."""
    dest = Path(dest)
    dest.unlink(missing_ok=True)

    opts = {
        "outtmpl": str(dest.with_suffix("")) + ".%(ext)s",
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Instagram in particular blocks unauthenticated clients unless we look like a browser
        "http_headers": {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                                       "Chrome/122.0 Safari/537.36"},
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as e:
        msg = str(e)
        low = msg.lower()
        if "login" in low or "cookies" in low or "rate-limit" in low:
            raise FetchError(
                "That post wants a login (Instagram usually does). Download the video "
                "manually and drop the file in instead."
            ) from e
        if "not available" in low or "unavailable" in low or "404" in low:
            raise FetchError("That video isn't available — check the link is public.") from e
        # yt-dlp errors trail a 'please report this issue' banner; keep only the useful part
        clean = msg.splitlines()[0].split("; please report")[0].replace("ERROR: ", "")
        raise FetchError(f"Couldn't download that: {clean}") from e

    if not dest.exists():
        # yt-dlp may have written a different extension despite merge_output_format
        candidates = sorted(dest.parent.glob(dest.stem + ".*"))
        candidates = [c for c in candidates if c.suffix in (".mp4", ".mkv", ".webm", ".mov")]
        if not candidates:
            raise FetchError("Download finished but no video file appeared.")
        candidates[0].rename(dest)

    return {
        "title": (info.get("title") or "")[:80],
        "uploader": info.get("uploader") or info.get("channel") or "",
        "duration": info.get("duration"),
        "path": str(dest),
    }
