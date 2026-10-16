{{- define "ppe-compliance-monitor.name" -}}
{{- .Chart.Name -}}
{{- end -}}

{{- define "ppe-compliance-monitor.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "ppe-compliance-monitor.labels" -}}
app.kubernetes.io/name: {{ include "ppe-compliance-monitor.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end -}}

{{- define "ppe-compliance-monitor.storageFullname" -}}
{{- $storage := index .Values "aws-compatible-storage" | default dict -}}
{{- if $storage.fullnameOverride -}}
{{- $storage.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := $storage.nameOverride | default "aws-compatible-storage" -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "ppe-compliance-monitor.storageSecretName" -}}
{{- $storage := index .Values "aws-compatible-storage" | default dict -}}
{{- $s3 := $storage.s3 | default dict -}}
{{- if $s3.existingSecret -}}
{{- $s3.existingSecret -}}
{{- else -}}
{{- printf "%s-credentials" (include "ppe-compliance-monitor.storageFullname" .) -}}
{{- end -}}
{{- end -}}

{{- define "ppe-compliance-monitor.s3Env" -}}
{{- $storage := index .Values "aws-compatible-storage" | default dict -}}
{{- $s3 := $storage.s3 | default dict -}}
- name: AWS_ACCESS_KEY_ID
  valueFrom:
    secretKeyRef:
      name: {{ include "ppe-compliance-monitor.storageSecretName" . }}
      key: AWS_ACCESS_KEY_ID
- name: AWS_SECRET_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "ppe-compliance-monitor.storageSecretName" . }}
      key: AWS_SECRET_ACCESS_KEY
- name: AWS_ENDPOINT_URL
  value: {{ printf "http://%s:%v" (include "ppe-compliance-monitor.storageFullname" .) $storage.service.s3Port | quote }}
- name: AWS_DEFAULT_REGION
  value: {{ $s3.region | default "us-east-1" | quote }}
{{- end -}}
