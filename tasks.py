"""
Pyinvoke tasks.py file for automating releases and admin stuff.

To cut a new pymatgen release:

    invoke update-changelog
    git commit -am "Update changelog"
    invoke release
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import requests
from invoke import task

if TYPE_CHECKING:
    from invoke import Context


def _is_breaking_pr(pr: dict) -> bool:
    """A PR is breaking if its title starts with the ``[breaking]`` prefix."""
    return bool(re.match(r"^\[breaking\]\s*", pr.get("title", "").strip(), flags=re.IGNORECASE))


def _extract_breaking_notes(pr: dict) -> str:
    """Extract the ``## Breaking Changes`` section of a PR body, if present.

    The section runs until the next heading at or above its own level, so
    ``###`` subsections (e.g. ``### Migration``) are kept; leftover HTML
    comments (e.g. unfilled template placeholders) are stripped.
    """
    lines = (pr.get("body") or "").split("\n")
    start = None
    level = 2
    for idx, line in enumerate(lines):
        if m := re.match(r"^(#{2,4})\s+breaking changes\s*$", line, flags=re.IGNORECASE):
            start = idx
            level = len(m.group(1))
            break
    if start is None:
        return ""
    notes = []
    for line in lines[start + 1 :]:
        if re.match(rf"^#{{1,{level}}}\s", line):  # heading at/above the section level ends it
            break
        notes.append(line)
    notes = re.sub(r"<!--.*?-->", "", "\n".join(notes), flags=re.DOTALL)
    return notes.strip()


def _compatibility_section(version: str, breaking_prs: list[dict]) -> str:
    """Format the ``### v<version>`` section of COMPATIBILITY.md for ``[breaking]``-titled PRs."""
    if not version:
        raise ValueError("version must be provided")
    lines = [f"### v{version}", ""]
    for pr in breaking_prs:
        num = pr["number"]
        title = pr["title"].strip()
        author = pr["user"]["login"]
        url = f"https://github.com/materialsproject/pymatgen-core/pull/{num}"
        lines.append(f"* {title} by @{author} in [#{num}]({url})")
        if notes := _extract_breaking_notes(pr):
            lines.append("")
            lines.extend(f"  {line}" for line in notes.split("\n"))
        lines.append("")
    return "\n".join(lines)


CHANGELOG_PROMPT = """
Provide a concise summary of the following pull requests as a change log for the pymatgen package. Format the summary
as a markdown bulleted list. Make sure to include the GitHub ids of all the authors. Do not include any code
blocks and timing outputs. Do not include any dependabot and pre-commit PRs.
"""


@task
def release_github(ctx: Context, version: str) -> None:
    """
    Release to Github using Github API.

    Args:
        ctx (Context): The context.
        version (str): The version.
    """
    with open("CHANGES.md", encoding="utf-8") as file:
        contents = file.read()
    tokens = re.split(r"\n\#\#\s", contents)
    desc = tokens[1].strip()
    tokens = desc.split("\n")
    desc = "\n".join(tokens[1:]).strip()
    payload = {
        "tag_name": f"v{version}",
        "name": f"v{version}",
        "body": desc,
        "draft": False,
        "prerelease": False,
    }
    response = requests.post(
        "https://api.github.com/repos/materialsproject/pymatgen-core/releases",
        data=json.dumps(payload),
        headers={"Authorization": f"token {os.environ['GITHUB_RELEASES_TOKEN']}"},
        timeout=60,
    )
    response.raise_for_status()
    print(response.text)


@task
def update_changelog(ctx: Context, version: str | None = None, dry_run: bool = False) -> None:
    """Create a preliminary change log using the git logs.

    Args:
        ctx (invoke.Context): The context object.
        version (str, optional): The version to use for the change log. If not provided, it will
            use the current date in the format 'YYYY.M.D'. Defaults to None.
        dry_run (bool, optional): If True, the function will only print the changes without
            updating the actual change log file. Defaults to False.
    """
    version = version or f"{datetime.now(tz=UTC):%Y.%-m.%-d}"
    last_tag = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], text=True).strip()
    print(f"Getting all commits since {last_tag}")
    output = subprocess.check_output(["git", "log", "--pretty=format:%s", f"{last_tag}..HEAD"])
    lines = []
    breaking_prs = []
    ignored_commits = []
    for line in output.decode("utf-8").strip().split("\n"):
        re_match = re.match(r".*\(\#(\d+)\)", line)
        if re_match and "materialsproject/dependabot/pip" not in line:
            pr_number = re_match[1].strip()
            headers = {}
            if token := os.getenv("GITHUB_ACCESS_TOKEN"):
                headers = {
                    "Accept": "application/vnd.github.v3+json",  # Recommended for GitHub API
                    "Authorization": f"token {token}",
                }

            response = requests.get(
                f"https://api.github.com/repos/materialsproject/pymatgen-core/pulls/{pr_number}",
                headers=headers,
                timeout=60,
            )
            resp = response.json()
            lines += [f"- PR #{pr_number} {resp['title'].strip()} by @{resp['user']['login']}"]
            if _is_breaking_pr(resp):
                breaking_prs.append(resp)
            if body := resp["body"]:
                for ll in map(str.strip, body.split("\n")):
                    if ll in ("", "## Summary"):
                        continue
                    if ll.startswith(("## Checklist", "## TODO")):
                        break
                    lines += [f"    {ll}"]
        else:
            ignored_commits += [line]

    body = "\n".join(lines)
    try:
        # Use OpenAI to improve changelog. Requires openai to be installed and an OPENAPI_KEY env variable.
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAPI_KEY"])

        messages = [{"role": "user", "content": CHANGELOG_PROMPT + f": '{body}'"}]
        chat = client.chat.completions.create(model="gpt-5", messages=messages)

        reply = chat.choices[0].message.content
        body = "\n".join(reply.split("\n")[1:-1])
        body = body.strip().strip("`")
        print(f"ChatGPT Summary of Changes:\n{body}")

    except BaseException as ex:
        print(f"Unable to use openai due to {ex}")
    with open("CHANGES.md", encoding="utf-8") as file:
        contents = file.read()
    delim = "##"
    tokens = contents.split(delim)
    tokens.insert(1, f"## v{version}\n\n{body}\n\n")
    if dry_run:
        print(tokens[0] + "##".join(tokens[1:]))
    else:
        with open("CHANGES.md", mode="w", encoding="utf-8") as file:
            file.write(tokens[0] + "##".join(tokens[1:]))
        ctx.run("open CHANGES.md")

    if breaking_prs:
        compat = _compatibility_section(version, breaking_prs)
        with open("COMPATIBILITY.md", encoding="utf-8") as file:
            contents = file.read()
        marker = "## Recent Breaking Changes"
        if marker not in contents:
            raise RuntimeError(f"{marker!r} section not found in COMPATIBILITY.md")
        idx = contents.index(marker) + len(marker)
        new_contents = contents[:idx] + "\n\n" + compat + contents[idx:]
        if dry_run:
            print(new_contents)
        else:
            with open("COMPATIBILITY.md", mode="w", encoding="utf-8") as file:
                file.write(new_contents)
        print(f"Logged {len(breaking_prs)} breaking change(s) in COMPATIBILITY.md for v{version}.")
    else:
        print("No `[breaking]`-titled PRs since last tag; COMPATIBILITY.md not updated.")

    print("The following commit messages were not included...")
    print("\n".join(ignored_commits))


@task
def release(ctx: Context, version: str | None = None) -> None:
    """
    Run full sequence for releasing pymatgen.

    Docs are built downstream by the pymatgen repo (triggered via repository_dispatch
    from the Release workflow), so there is no doc-generation step here.

    Args:
        ctx (invoke.Context): The context object.
        version (str, optional): The version to release.
    """
    version = version or f"{datetime.now(tz=UTC):%Y.%-m.%-d}"
    if not re.fullmatch(r"\d{4}\.\d{1,2}\.\d{1,2}", version):
        raise ValueError("Version must use the YYYY.M.D calendar format")

    tag = f"v{version}"
    if subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout:
        raise RuntimeError("Commit or stash all changes before creating a release tag")
    if (
        subprocess.run(
            ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"], check=False, capture_output=True
        ).returncode
        == 0
    ):
        raise RuntimeError(f"Tag {tag} already exists")

    ctx.run(f"git tag {tag}")
    ctx.run(f"git push origin {tag}")
    release_github(ctx, version)


@task
def lint(ctx: Context) -> None:
    """
    Run linting tools.

    Args:
        ctx (invoke.Context): The context object.
    """
    for cmd in ("ruff", "mypy", "ruff format"):
        ctx.run(f"{cmd} pymatgen")
