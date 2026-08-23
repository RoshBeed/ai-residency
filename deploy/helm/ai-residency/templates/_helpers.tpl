{{- define "ai-residency-hub.labels" -}}
app.kubernetes.io/part-of: ai-residency-hub
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}
