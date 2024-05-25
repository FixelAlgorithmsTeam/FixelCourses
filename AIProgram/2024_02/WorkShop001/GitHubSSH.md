# Github SSH on Remote / Mac

Setting up SSH:

1. connect to server

2. create key `ssh-keygen -t ed25519`

3. copy key `cat ~/.ssh/id_ed25519.pub`

should be something like  
`ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPXD3sdabbxNu/7KmhWn0o4+xUQL90V2GS7LljZzGbzH ubuntu@ip-172-31-26-222`

4. Login to Github -> Settings -> SSH and GPG Keys -> `New SSH key`
