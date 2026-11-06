# Github SSH on Remote / Mac

[![Visitors](https://img.shields.io/badge/Fixel-Algorithms-blue)](https://fixelalgorithms.gitlab.io)
[![Visitors](https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fgithub.com%2FRoyiAvital%2FStackExchangeCodes&labelColor=%23f47373&countColor=%23555555&style=plastic)](https://github.com/FixelAlgorithmsTeam/FixelCourses) <!-- https://www.visitorbadge.io -->

Setting up SSH:

1. connect to server

2. create key `ssh-keygen -t ed25519`

3. copy key `cat ~/.ssh/id_ed25519.pub`

should be something like  
`ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPXD3sdabbxNu/7KmhWn0o4+xUQL90V2GS7LljZzGbzH ubuntu@ip-172-31-26-222`

4. Login to Github -> Settings -> SSH and GPG Keys -> `New SSH key`
