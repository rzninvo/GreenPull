import logging

from github import Github, Auth, GithubException

logger = logging.getLogger("greenpull")


class GitHubService:
    """Wrapper around PyGithub for creating optimization PRs."""

    def __init__(self, token: str, repo_url: str):
        """
        Args:
            token: GitHub Personal Access Token
            repo_url: Full GitHub URL (https://github.com/owner/repo) or "owner/repo"
        """
        auth = Auth.Token(token)
        self.g = Github(auth=auth)
        self.repo_slug = self._normalize_repo(repo_url)
        self.repo = self.g.get_repo(self.repo_slug)

    @staticmethod
    def _normalize_repo(url: str) -> str:
        """Convert https://github.com/owner/repo to owner/repo."""
        url = url.rstrip("/")
        if "github.com/" in url:
            parts = url.split("github.com/")[1]
            for suffix in ("/tree/", "/blob/", "/pull/", "/issues/"):
                if suffix in parts:
                    parts = parts.split(suffix)[0]
            return parts.rstrip("/").removesuffix(".git")
        return url

    def create_branch(self, branch_name: str, base_branch: str = "main") -> None:
        """Create a new branch from the base branch."""
        base_ref = self.repo.get_git_ref(f"heads/{base_branch}")
        base_sha = base_ref.object.sha
        try:
            self.repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_sha)
            logger.info(f"[GitHub] Created branch '{branch_name}' from '{base_branch}'")
        except GithubException as e:
            if e.status == 422:
                logger.info(f"[GitHub] Branch '{branch_name}' already exists")
            else:
                raise

    def update_file(
        self, file_path: str, content: str, commit_message: str, branch_name: str
    ) -> dict:
        """Create or update a file on a branch."""
        try:
            existing = self.repo.get_contents(file_path, ref=branch_name)
            result = self.repo.update_file(
                path=file_path,
                message=commit_message,
                content=content,
                sha=existing.sha,
                branch=branch_name,
            )
        except GithubException:
            result = self.repo.create_file(
                path=file_path,
                message=commit_message,
                content=content,
                branch=branch_name,
            )
        return {"commit_sha": result["commit"].sha}

    def create_multi_file_optimization_pr(
        self,
        file_patches: list[dict],
        title: str,
        body: str,
        branch_name: str = "greenpull/optimization",
        base_branch: str = "main",
    ) -> dict:
        """Full flow: create branch, push ALL patched files, open PR."""
        self.create_branch(branch_name, base_branch)
        for i, fp in enumerate(file_patches):
            msg = f"{title} ({i+1}/{len(file_patches)}: {fp['file_path']})" if len(file_patches) > 1 else title
            self.update_file(
                file_path=fp["file_path"],
                content=fp["patched_code"],
                commit_message=msg,
                branch_name=branch_name,
            )
        pr = self.repo.create_pull(base=base_branch, head=branch_name, title=title, body=body)
        logger.info(f"[GitHub] Created multi-file PR #{pr.number}: {pr.html_url}")
        return {"number": pr.number, "url": pr.html_url}

    def create_optimization_pr(
        self,
        file_path: str,
        patched_code: str,
        title: str,
        body: str,
        branch_name: str = "greenpull/optimization",
        base_branch: str = "main",
    ) -> dict:
        """Full flow: create branch, push patched file, open PR."""
        self.create_branch(branch_name, base_branch)
        self.update_file(
            file_path=file_path,
            content=patched_code,
            commit_message=title,
            branch_name=branch_name,
        )
        pr = self.repo.create_pull(base=base_branch, head=branch_name, title=title, body=body)
        logger.info(f"[GitHub] Created PR #{pr.number}: {pr.html_url}")
        return {
            "number": pr.number,
            "url": pr.html_url,
        }
