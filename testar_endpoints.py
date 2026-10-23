import urllib.request, json

BASE = 'http://localhost:8000'

def post(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f'{BASE}{path}', data=body,
        headers={'Content-Type': 'application/json'}, method='POST'
    )
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())

def get(path):
    req = urllib.request.urlopen(f'{BASE}{path}')
    return json.loads(req.read())

# ── TESTE 4: Leitura de Alerta ────────────────────────────────────────
leitura_alerta = {
    'temperatura_fuso': 145.0, 'vibracao_x': 1.0, 'vibracao_y': 0.8,
    'vibracao_z': 1.0, 'corrente_motor': 16.0, 'rpm_fuso': 1500.0,
    'pressao_refrigerante': 4.5, 'desgaste_ferramenta': 60.0
}
print('=' * 60)
print('TESTE 4 - POST /analisar (leitura ALERTA)')
print('=' * 60)
r = post('/analisar', leitura_alerta)
print(json.dumps(r, indent=2, ensure_ascii=False))

# ── TESTE 5: Simulação ────────────────────────────────────────────────
print()
print('=' * 60)
print('TESTE 5 - POST /simular (3 leituras)')
print('=' * 60)
r = post('/simular', {'n_amostras': 3, 'seed': 42})
for i, l in enumerate(r['leituras'], 1):
    temp = l['temperatura_fuso']
    vib  = l['vibracao_x']
    desg = l['desgaste_ferramenta']
    print(f'  Leitura {i}: temp={temp:.1f}C | vib_x={vib:.2f}mm/s | desgaste={desg:.1f}%')

# ── TESTE 6: Relatório ────────────────────────────────────────────────
print()
print('=' * 60)
print('TESTE 6 - GET /relatorio (historico de analises)')
print('=' * 60)
r = get('/relatorio')
print(f'Total de analises registradas: {r["total_analises"]}')
for entry in r['historico']:
    ts   = entry['timestamp'][:19]
    risk = entry['nivel_risco']
    prob = entry['probabilidade_falha']
    acoes = entry['acoes_executadas']
    print(f'  [{ts}] Risco: {risk} | Prob.falha: {prob:.0%} | Acoes: {acoes}')
