import paramiko


class QuestConnection:
    """
    # Summary
        This class is used to connect to Quest and execute commands.
        
    ## Attributes
        - `ssh`: paramiko.SSHClient
        - `hostname`: str, default 'quest.northwestern.edu'
        - `username`: str, this is the username for Quest and is the same as your NetID
        - `password`: str, this is the password for Quest and is the same as your NetID password
    """
    def __init__(self, username, password) -> None:

        self.ssh = paramiko.SSHClient()
        self.hostname = 'quest.northwestern.edu'
        self.username = username
        self.password = password

    def connect_ssh(self):
        """
        # Summary
            This method is used to connect to Quest using SSH.
        
        """
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            self.ssh.connect(hostname=self.hostname,
                             username=self.username,
                             password=self.password
                             )
            if self.ssh.get_transport().is_active():
                print("Connected to Quest")
            else:
                print("Failed to connect to Quest")

        except paramiko.AuthenticationException:
            print("Authentication failed, please verify username and password")

        except paramiko.SSHException:
            print("Unable to establish SSH connection")

        except paramiko.Exception as e:
            print("Error: " + str(e))

        finally:
            self.ssh.close()

    def close(self):
        self.ssh.close()
        print("Quest connection closed")

    # def execute_command(self, command):
    #     stdin, stdout, stderr = self.ssh.exec_command(command)
    #     return stdout.readlines()


if __name__ == "__main__":
    quest = QuestConnection('mds8301', 'Ms386874!!!')
    quest.connect_ssh()
    quest.execute_command('ls')
    quest.close()
