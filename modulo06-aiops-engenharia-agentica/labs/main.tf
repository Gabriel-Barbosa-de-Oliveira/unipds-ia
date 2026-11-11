terraform {
  required_version = ">= 1.0.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

resource "aws_s3_bucket" "nexus_apollo_data" {
  bucket = "nexus-apollo-data"
  force_destroy = false

  # Enable server-side encryption (AES-256)
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  # Enable versioning
  versioning {
    enabled = true
  }

  # Block all public access
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true

  tags = {
    Name        = "nexus-apollo-data"
    Environment = "production"
    ManagedBy   = "Terraform"
  }
}

# Optional: bucket policy that denies any non‑TLS requests
resource "aws_s3_bucket_policy" "nexus_apollo_data_policy" {
  bucket = aws_s3_bucket.nexus_apollo_data.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Sid       = "DenyInsecureTransport",
        Effect    = "Deny",
        Principal = "*",
        Action    = "s3:*",
        Resource = [
          "${aws_s3_bucket.nexus_apollo_data.arn}",
          "${aws_s3_bucket.nexus_apollo_data.arn}/*"
        ],
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })
}