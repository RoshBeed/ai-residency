{{- define "ai-residency.labels" -}}
app.kubernetes.io/part-of: ai-residency
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{/*
The pod- and container-level security contexts, defined once so backends, the frontend,
Redis, the init containers and the training Jobs cannot drift apart. Each takes the root
context as its argument.
*/}}
{{- define "ai-residency.podSecurityContext" -}}
{{- if .Values.securityContext.enabled }}
securityContext:
  {{- toYaml .Values.securityContext.pod | nindent 2 }}
{{- end }}
{{- end -}}

{{- define "ai-residency.containerSecurityContext" -}}
{{- if .Values.securityContext.enabled }}
securityContext:
  {{- toYaml .Values.securityContext.container | nindent 2 }}
{{- end }}
{{- end -}}

{{/*
Resource requests and limits for one container. Takes a dict of {ctx, spec}, where spec
is a backend entry or the frontend/redis block.

No CPU limit is emitted anywhere, deliberately: a CPU limit throttles rather than kills,
and throttling a model mid-inference makes a fast request slow for no benefit here.
*/}}
{{- define "ai-residency.resources" -}}
{{- $spec := .spec -}}
{{- if or $spec.cpuRequest $spec.memoryRequest $spec.memoryLimit }}
resources:
  {{- if or $spec.cpuRequest $spec.memoryRequest }}
  requests:
    {{- if $spec.cpuRequest }}
    cpu: {{ $spec.cpuRequest }}
    {{- end }}
    {{- if $spec.memoryRequest }}
    memory: {{ $spec.memoryRequest }}
    {{- end }}
  {{- end }}
  {{- if $spec.memoryLimit }}
  limits:
    memory: {{ $spec.memoryLimit }}
  {{- end }}
{{- end }}
{{- end -}}

{{/*
One entry of a pod's `volumes:` list. Takes {ctx, vol}.

A volume with `kind: hfCache` is resolved through .Values.hfCache, so the single-node
hostPath default can be swapped for a PersistentVolumeClaim -- or dropped entirely --
without editing every service that mounts the cache. Anything else is taken literally.
*/}}
{{- define "ai-residency.volumeSource" -}}
{{- $ctx := .ctx -}}
{{- $vol := .vol -}}
{{- if eq ($vol.kind | default "") "hfCache" -}}
{{- if eq $ctx.Values.hfCache.mode "pvc" -}}
persistentVolumeClaim:
  claimName: {{ required "hfCache.claimName is required when hfCache.mode is pvc" $ctx.Values.hfCache.claimName }}
{{- else if eq $ctx.Values.hfCache.mode "hostPath" -}}
hostPath:
  path: {{ required "hfCache.hostPath is required when hfCache.mode is hostPath" (tpl $ctx.Values.hfCache.hostPath $ctx) }}
  type: DirectoryOrCreate
{{- else -}}
emptyDir: {}
{{- end }}
{{- else -}}
hostPath:
  path: {{ tpl $vol.hostPath $ctx }}
  type: {{ $vol.type }}
{{- end }}
{{- end -}}

{{/*
An image reference, by digest when the manifest has one.

`deploy/images.lock` names the exact image each service runs, and `pin_images.py`
propagates it here as a per-service `digest`. A digest cannot be moved, and it is
per-image -- which a single shared `image.tag` could not be. `image.useDigest=false` falls back to the tag, which is what a local cluster needs:
images built on this machine have their own digests, and published amd64 images cannot
run on Docker Desktop's Kubernetes node at all -- that node is arm64 on Apple Silicon.

Takes a dict: `spec` (the service's own values) and `ctx` (the root context).
*/}}
{{- define "ai-residency.image" -}}
{{- $spec := .spec -}}
{{- $ctx := .ctx -}}
{{- if and $spec.digest $ctx.Values.image.useDigest -}}
{{ $spec.image }}@{{ $spec.digest }}
{{- else -}}
{{ $spec.image }}:{{ $ctx.Values.image.tag }}
{{- end -}}
{{- end -}}
