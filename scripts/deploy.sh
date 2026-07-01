#!/bin/bash
# deploy.sh — Deploy rag-from-scratch to AWS EC2 (t2.micro, free tier)
#
# Prerequisites:
#   - AWS CLI configured: aws configure
#   - Docker installed locally
#   - A key pair created in AWS Console (EC2 -> Key Pairs)
#
# Usage:
#   chmod +x scripts/deploy.sh
#   ./scripts/deploy.sh
#
# The script:
#   1. Creates a security group allowing HTTP (8000) and SSH (22)
#   2. Launches a t2.micro EC2 instance with Amazon Linux 2
#   3. SSHs in, installs Docker, pulls the image, and runs the container
#
# To stop and avoid charges:
#   aws ec2 terminate-instances --instance-ids <instance-id>

set -e

REGION="us-east-1"
AMI_ID="ami-0c02fb55956c7d316"  # Amazon Linux 2 in us-east-1
INSTANCE_TYPE="t2.micro"
KEY_NAME="rag-key"              # Change to your key pair name
PORT=8000

echo "==> Creating security group..."
SG_ID=$(aws ec2 create-security-group \
    --group-name rag-from-scratch-sg \
    --description "RAG server security group" \
    --region $REGION \
    --query 'GroupId' \
    --output text 2>/dev/null || \
    aws ec2 describe-security-groups \
        --group-names rag-from-scratch-sg \
        --region $REGION \
        --query 'SecurityGroups[0].GroupId' \
        --output text)

echo "Security group: $SG_ID"

echo "==> Opening ports 22 (SSH) and $PORT (HTTP)..."
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp --port 22 --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || true

aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp --port $PORT --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || true

echo "==> Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --region $REGION \
    --user-data '#!/bin/bash
        yum update -y
        yum install -y docker
        systemctl start docker
        systemctl enable docker
        usermod -aG docker ec2-user' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance ID: $INSTANCE_ID"
echo "==> Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "==> Instance running at $PUBLIC_IP"
echo ""
echo "==> Next steps (run after ~60s for Docker to finish installing):"
echo ""
echo "  # SSH into the instance"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@$PUBLIC_IP"
echo ""
echo "  # On the instance, pull and run the container"
echo "  docker pull ghcr.io/sujanuj/rag-from-scratch:latest"
echo "  docker run -d -p $PORT:$PORT -e HF_API_KEY=your_key ghcr.io/sujanuj/rag-from-scratch:latest"
echo ""
echo "  # Test the server"
echo "  curl http://$PUBLIC_IP:$PORT/health"
echo ""
echo "==> To terminate the instance and avoid charges:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
