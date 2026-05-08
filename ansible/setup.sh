#!/bin/bash
set -e

if [ ! -f "ansible/.vault_pass" ]; then
    echo "Creating vault password file..."
    openssl rand -base64 24 > ansible/.vault_pass
    chmod 600 ansible/.vault_pass
    echo "Vault password file created at ansible/.vault_pass"
fi
