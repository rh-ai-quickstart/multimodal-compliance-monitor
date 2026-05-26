{{- define "jupyter-training.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "jupyter-training.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "jupyter-training.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "jupyter-training.selectorLabels" -}}
app.kubernetes.io/name: {{ include "jupyter-training.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "jupyter-training.labels" -}}
helm.sh/chart: {{ include "jupyter-training.chart" . }}
{{ include "jupyter-training.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "jupyter-training.jupyterServerConfigJSON" -}}
{{- $gitOff := not .Values.jupyterGit.enabled -}}
{"ServerApp":{"ip":"0.0.0.0","allow_origin":"*","allow_remote_access":true,"trust_xheaders":true,"root_dir":{{ .Values.notebookRootDir | toJson }},"default_url":"/lab/tree/yolo_training.ipynb"{{ if $gitOff }},"jpserver_extensions":{"jupyterlab_git":false}{{ end }} }}
{{- end }}

{{- define "jupyter-training.jupyterLabConfigJSON" -}}
{{- if .Values.jupyterGit.enabled -}}
{"defaultUrl":"/lab/tree/yolo_training.ipynb"}
{{- else -}}
{"disabledExtensions":["@jupyterlab/git"],"defaultUrl":"/lab/tree/yolo_training.ipynb"}
{{- end -}}
{{- end }}

{{- define "jupyter-training.jupyterConfigChecksum" -}}
{{- printf "%s\n%s" (include "jupyter-training.jupyterServerConfigJSON" .) (include "jupyter-training.jupyterLabConfigJSON" .) | sha256sum -}}
{{- end }}
