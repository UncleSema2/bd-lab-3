.PHONY: decrypt-vault decrypt-dvc vault-setup

ANSIBLE_VAULT_PASS_FILE := ansible/.vault_pass

vault-setup:
	@bash ansible/setup.sh

decrypt-vault: $(ANSIBLE_VAULT_PASS_FILE)
	@vault_content=$$(ANSIBLE_VAULT_PASSWORD_FILE=$(ANSIBLE_VAULT_PASS_FILE) ansible-vault view ansible/group_vars/all/vault.yml) && \
	echo "$$vault_content" | grep '^cassandra_host:' | awk '{print "CASSANDRA_HOST=" $$2}' | sed 's/"//g' > .env && \
	echo "$$vault_content" | grep '^cassandra_port:' | awk '{print "CASSANDRA_PORT=" $$2}' | sed 's/"//g' >> .env && \
	echo "$$vault_content" | grep '^cassandra_keyspace:' | awk '{print "CASSANDRA_KEYSPACE=" $$2}' | sed 's/"//g' >> .env && \
	echo "$$vault_content" | grep '^cassandra_username:' | awk '{print "CASSANDRA_USERNAME=" $$2}' | sed 's/"//g' >> .env && \
	echo "$$vault_content" | grep '^cassandra_password:' | awk '{print "CASSANDRA_PASSWORD=" $$2}' | sed 's/"//g' >> .env && \
	echo "$$vault_content" | grep '^model_version:' | awk '{print "MODEL_VERSION=" $$2}' | sed 's/"//g' >> .env && \
	echo ".env generated from Ansible Vault"

decrypt-dvc: $(ANSIBLE_VAULT_PASS_FILE)
	@vault_content=$$(ANSIBLE_VAULT_PASSWORD_FILE=$(ANSIBLE_VAULT_PASS_FILE) ansible-vault view ansible/group_vars/all/vault.yml) && \
	aws_key=$$(echo "$$vault_content" | grep '^aws_access_key_id:' | awk '{print $$2}' | sed 's/"//g') && \
	aws_secret=$$(echo "$$vault_content" | grep '^aws_secret_access_key:' | awk '{print $$2}' | sed 's/"//g') && \
	dvc remote modify remote --local access_key_id "$$aws_key" && \
	dvc remote modify remote --local secret_access_key "$$aws_secret" && \
	echo ".dvc/config.local updated with DVC credentials from vault"

encrypt-vault:
	ANSIBLE_VAULT_PASSWORD_FILE=$(ANSIBLE_VAULT_PASS_FILE) ansible-vault encrypt ansible/group_vars/all/vault.yml
