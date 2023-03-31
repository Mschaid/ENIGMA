import paramiko


class QuestConnection:
    def __init__(self, username, password) -> None:
        self.ssh = paramiko.SSHClient()
        self.hostname = 'quest.northwestern.edu'
        self.username = username
        self.password = password

    def connect_ssh(self):
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
