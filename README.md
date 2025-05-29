docker build --no-cache -t aitutor-backend:latest . docker tag aitutor-backend:latest aitutoracrfullertonvatsal.azurecr.io/aitutor-backend:latest
docker push aitutoracrfullertonvatsal.azurecr.io/aitutor-backend:latest az webapp restart --name aitutorwebappfullertonvatsal --resource-group aitutorrg those are the actual commands
