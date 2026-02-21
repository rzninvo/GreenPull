from github import Github, Auth


def read_credentials(file_path="credentials.txt"):
    """
    Read GitHub personal access token from credentials.txt file
    Expected format:
    Line 1: Personal Access Token
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.read().strip().split('\n')
            if len(lines) >= 1:
                token = lines[0].strip()  # This should be your Personal Access Token
                return token
            else:
                print("Error: credentials.txt should have token on separate lines")
                return None
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
        return None, None
    except Exception as e:
        print(f"Error reading credentials: {e}")
        return None, None


class Repo: 
    def __init__(self, token, repo_url):

        auth = Auth.Token(token)
        self.g = Github(auth=auth)
        self.repo_url = repo_url
        self.repo = self.g.get_repo(repo_url)

    def getFile(self, file_path):
        """
        Get file content from GitHub repo

        Returns:
            str: File content or error message
        """
        try:
            
            # Get the file content
            file = self.repo.get_contents(file_path)
            content = file.decoded_content.decode('utf-8')
            
            return content
        
        except Exception as e:
            return f"Error: {e}"


    def updateFile(self, file_path, content, commit_message, branch_name="main"):
        """
        Create or update a file in the repository
        
        Args:
            file_path (str): Path to the file in the repo
            content (str): Content to write to the file
            commit_message (str): Commit message
            branch_name (str): Branch to commit to
        
        Returns:
            dict: Commit information or error message
        """
        try:
            # Check if file exists
            try:
                existing_file = self.repo.get_contents(file_path, ref=branch_name)
                # File exists, update it
                result = self.repo.update_file(
                    path=file_path,
                    message=commit_message,
                    content=content,
                    sha=existing_file.sha,
                    branch=branch_name
                )
                return {
                    "action": "updated",
                    "commit_sha": result["commit"].sha,
                    "commit_url": result["commit"].html_url
                }
            except:
                # File doesn't exist, create it
                result = self.repo.create_file(
                    path=file_path,
                    message=commit_message,
                    content=content,
                    branch=branch_name
                )
                return {
                    "action": "created",
                    "commit_sha": result["commit"].sha,
                    "commit_url": result["commit"].html_url
                }
        
        except Exception as e:
            return f"Error: {e}"

    def makePullRequest(self, base, head, title, body):
        """
        Create a pull request on GitHub
        
        Args:
            base (str): The branch you want to merge into (default: "main")
            head (str): The branch that contains your changes
            title (str): Title of the pull request
            body (str): Description of the pull request

        
        Returns:
            dict: Pull request information or error message
        """
        try:
            
            # Create the pull request
            pr = self.repo.create_pull(
                base=base,
                head=head,
                title=title,
                body=body
            )
            pr 
            
            return {
                "number": pr.number,
                "title": pr.title,
                "url": pr.html_url,
                "state": pr.state
            }
        
        except Exception as e:
            return f"Error: {e}"


if __name__ == "__main__":
    # Read credentials from credentials.txt 
    token = read_credentials()
    repo_url = "SABR007/greenpull_test_repo"
    file_path = "test_extracted_code.py"
    

    # Create a new Repo instance with auth token, repo_url 
    g = Repo(token, repo_url)


    # Use getFile with "file_path" to get contents of target file
    contents = g.getFile(file_path)


    # Example optimzed file
    optimized_file = contents + "\n OPTIMIZEDDDDD UEESSESSSS"

    # Step 1: Create a file with changes in the test branch
    print("Creating a test file in the 'test' branch...")
            
        # Args:
        #     file_path (str): Path to the file in the repo
        #     content (str): Content to write to the file
        #     commit_message (str): Commit message
        #     branch_name (str): Branch to commit to
        
    file_result = g.updateFile(
        file_path=file_path,
        content=optimized_file,
        commit_message="Code optimized",
        branch_name="test"
    )
    print("File creation result:", file_result)
    
    # Step 2: Now create the pull request (should work since test branch has commits)
    print("\nCreating pull request...")
    pr_result = g.makePullRequest("main", "test", "Optimize Code", "This PR adds a test file to demonstrate the pull request functionality.")
    print("PR result:", pr_result)


