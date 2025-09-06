#! /bin/sh -f

TOKEN=$(sed -n 's#.*://[^:]*:\([^@]*\)@github.com.*#\1#p' ~/.git-credentials | tail -n1)
git config --global credential.helper store
tmp=$(mktemp); grep -v 'github.com' ~/.git-credentials 2>/dev/null > "$tmp" || true
mv "$tmp" ~/.git-credentials
chmod 600 ~/.git-credentials
printf 'https://%s:%s@github.com\n' 'Coslate' "$TOKEN" >> ~/.git-credentials