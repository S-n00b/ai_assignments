# Enterprise LLMOps Infrastructure
terraform {
  required_version = ">= 1.0"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

# Configure providers
provider "kubernetes" {
  config_path = var.kubeconfig_path
}

provider "helm" {
  kubernetes {
    config_path = var.kubeconfig_path
  }
}

provider "aws" {
  region = var.aws_region
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# Variables
variable "kubeconfig_path" {
  description = "Path to kubeconfig file"
  type        = string
  default     = "~/.kube/config"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "gcp_project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "llmops-enterprise"
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "cluster_name" {
  description = "Kubernetes cluster name"
  type        = string
  default     = "llmops-cluster"
}

# Data sources
data "kubernetes_namespace" "llmops" {
  metadata {
    name = "llmops-enterprise"
  }
}

# Storage Classes
resource "kubernetes_storage_class" "fast_ssd" {
  metadata {
    name = "fast-ssd"
  }
  storage_provisioner = "kubernetes.io/aws-ebs"
  parameters = {
    type = "gp3"
    iops = "3000"
    throughput = "125"
  }
  reclaim_policy = "Retain"
}

resource "kubernetes_storage_class" "slow_hdd" {
  metadata {
    name = "slow-hdd"
  }
  storage_provisioner = "kubernetes.io/aws-ebs"
  parameters = {
    type = "sc1"
  }
  reclaim_policy = "Retain"
}

# ConfigMaps
resource "kubernetes_config_map" "prometheus_config" {
  metadata {
    name      = "prometheus-config"
    namespace = "llmops-enterprise"
  }

  data = {
    "prometheus.yml" = <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)

  - job_name: 'ollama-metrics'
    static_configs:
      - targets: ['ollama-service:11434']
    metrics_path: '/api/metrics'
    scrape_interval: 30s

  - job_name: 'chroma-metrics'
    static_configs:
      - targets: ['chroma-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'langfuse-metrics'
    static_configs:
      - targets: ['langfuse-service:3000']
    metrics_path: '/api/metrics'
    scrape_interval: 30s

  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
      - role: node
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
      - target_label: __address__
        replacement: kubernetes.default.svc:443
      - source_labels: [__meta_kubernetes_node_name]
        regex: (.+)
        target_label: __metrics_path__
        replacement: /api/v1/nodes/${1}/proxy/metrics

  - job_name: 'kubernetes-cadvisor'
    kubernetes_sd_configs:
      - role: node
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
      - target_label: __address__
        replacement: kubernetes.default.svc:443
      - source_labels: [__meta_kubernetes_node_name]
        regex: (.+)
        target_label: __metrics_path__
        replacement: /api/v1/nodes/${1}/proxy/metrics/cadvisor

  - job_name: 'kubernetes-service-endpoints'
    kubernetes_sd_configs:
      - role: endpoints
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_scheme]
        action: replace
        target_label: __scheme__
        regex: (https?)
      - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_service_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
      - action: labelmap
        regex: __meta_kubernetes_service_label_(.+)
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_service_name]
        action: replace
        target_label: kubernetes_name
EOF
  }
}

resource "kubernetes_config_map" "grafana_datasources" {
  metadata {
    name      = "grafana-datasources"
    namespace = "llmops-enterprise"
  }

  data = {
    "prometheus.yml" = <<EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    url: http://prometheus-service:9090
    access: proxy
    isDefault: true
    editable: true
    jsonData:
      timeInterval: "5s"
      queryTimeout: "60s"
      httpMethod: "POST"

  - name: Ollama Metrics
    type: prometheus
    url: http://ollama-service:11434
    access: proxy
    editable: true
    jsonData:
      timeInterval: "30s"

  - name: Chroma Metrics
    type: prometheus
    url: http://chroma-service:8000
    access: proxy
    editable: true
    jsonData:
      timeInterval: "30s"

  - name: LangFuse Metrics
    type: prometheus
    url: http://langfuse-service:3000
    access: proxy
    editable: true
    jsonData:
      timeInterval: "30s"
EOF
  }
}

# Helm Releases
resource "helm_release" "nginx_ingress" {
  name       = "nginx-ingress"
  repository = "https://kubernetes.github.io/ingress-nginx"
  chart      = "ingress-nginx"
  version    = "4.8.3"
  namespace  = "ingress-nginx"

  create_namespace = true

  set {
    name  = "controller.service.type"
    value = "LoadBalancer"
  }

  set {
    name  = "controller.service.annotations.service\\.beta\\.kubernetes\\.io/aws-load-balancer-type"
    value = "nlb"
  }
}

resource "helm_release" "cert_manager" {
  name       = "cert-manager"
  repository = "https://charts.jetstack.io"
  chart      = "cert-manager"
  version    = "v1.13.0"
  namespace  = "cert-manager"

  create_namespace = true

  set {
    name  = "installCRDs"
    value = "true"
  }
}

resource "helm_release" "kube_prometheus_stack" {
  name       = "kube-prometheus-stack"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  version    = "45.7.1"
  namespace  = "llmops-enterprise"

  values = [
    <<EOF
prometheus:
  prometheusSpec:
    retention: 30d
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: fast-ssd
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 50Gi

grafana:
  enabled: true
  adminPassword: "admin123"
  persistence:
    enabled: true
    storageClassName: fast-ssd
    size: 10Gi
  grafana.ini:
    server:
      root_url: "http://grafana.llmops.local"
    security:
      admin_user: admin
      admin_password: admin123

alertmanager:
  enabled: true
  alertmanagerSpec:
    storage:
      volumeClaimTemplate:
        spec:
          storageClassName: fast-ssd
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 10Gi

kubeStateMetrics:
  enabled: true

nodeExporter:
  enabled: true

kubelet:
  enabled: true

coredns:
  enabled: true

kubeProxy:
  enabled: true

kubeEtcd:
  enabled: true

kubeScheduler:
  enabled: true

kubeControllerManager:
  enabled: true

kubeApiServer:
  enabled: true
EOF
  ]

  depends_on = [
    kubernetes_storage_class.fast_ssd,
    kubernetes_storage_class.slow_hdd
  ]
}

# Ingress Resources
resource "kubernetes_ingress_v1" "llmops_ingress" {
  metadata {
    name      = "llmops-ingress"
    namespace = "llmops-enterprise"
    annotations = {
      "kubernetes.io/ingress.class"                = "nginx"
      "cert-manager.io/cluster-issuer"             = "letsencrypt-prod"
      "nginx.ingress.kubernetes.io/ssl-redirect"   = "true"
      "nginx.ingress.kubernetes.io/force-ssl-redirect" = "true"
    }
  }

  spec {
    tls {
      hosts       = ["llmops.local", "grafana.llmops.local", "langfuse.llmops.local"]
      secret_name = "llmops-tls"
    }

    rule {
      host = "llmops.local"
      http {
        path {
          path      = "/"
          path_type = "Prefix"
          backend {
            service {
              name = "llmops-frontend-service"
              port {
                number = 8080
              }
            }
          }
        }
      }
    }

    rule {
      host = "grafana.llmops.local"
      http {
        path {
          path      = "/"
          path_type = "Prefix"
          backend {
            service {
              name = "grafana-service"
              port {
                number = 3000
              }
            }
          }
        }
      }
    }

    rule {
      host = "langfuse.llmops.local"
      http {
        path {
          path      = "/"
          path_type = "Prefix"
          backend {
            service {
              name = "langfuse-service"
              port {
                number = 3000
              }
            }
          }
        }
      }
    }
  }

  depends_on = [
    helm_release.nginx_ingress,
    helm_release.cert_manager
  ]
}

# ClusterIssuer for Let's Encrypt
resource "kubernetes_manifest" "letsencrypt_cluster_issuer" {
  manifest = {
    apiVersion = "cert-manager.io/v1"
    kind       = "ClusterIssuer"
    metadata = {
      name = "letsencrypt-prod"
    }
    spec = {
      acme = {
        server = "https://acme-v02.api.letsencrypt.org/directory"
        email  = "admin@llmops.local"
        privateKeySecretRef = {
          name = "letsencrypt-prod"
        }
        solvers = [
          {
            http01 = {
              ingress = {
                class = "nginx"
              }
            }
          }
        ]
      }
    }
  }

  depends_on = [
    helm_release.cert_manager
  ]
}

# Network Policies
resource "kubernetes_network_policy" "llmops_network_policy" {
  metadata {
    name      = "llmops-network-policy"
    namespace = "llmops-enterprise"
  }

  spec {
    pod_selector {
      match_labels = {
        app = "llmops"
      }
    }

    policy_types = ["Ingress", "Egress"]

    ingress {
      from {
        namespace_selector {
          match_labels = {
            name = "llmops-enterprise"
          }
        }
      }
    }

    ingress {
      ports {
        port     = "8080"
        protocol = "TCP"
      }
    }

    egress {
      to {
        namespace_selector {
          match_labels = {
            name = "llmops-enterprise"
          }
        }
      }
    }

    egress {
      ports {
        port     = "53"
        protocol = "UDP"
      }
      ports {
        port     = "53"
        protocol = "TCP"
      }
    }
  }
}

# Outputs
output "cluster_endpoint" {
  description = "Kubernetes cluster endpoint"
  value       = var.cluster_name
}

output "ingress_host" {
  description = "Ingress host for LLMOps"
  value       = "llmops.local"
}

output "grafana_url" {
  description = "Grafana URL"
  value       = "http://grafana.llmops.local"
}

output "prometheus_url" {
  description = "Prometheus URL"
  value       = "http://prometheus-service.llmops-enterprise.svc.cluster.local:9090"
}

output "langfuse_url" {
  description = "LangFuse URL"
  value       = "http://langfuse.llmops.local"
}
