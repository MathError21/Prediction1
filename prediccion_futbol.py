
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression

# === TUS CLAVES DE API ===
ODDS_API_KEY = "TU_CLAVE_DE_THE_ODDS_API"
API_FOOTBALL_KEY = "TU_CLAVE_DE_API_FOOTBALL"
LEAGUE_ID = 39      # Premier League (puedes cambiarlo)
SEASON = 2023       # Temporada a analizar

# === 1. Obtener mejor partido por cuotas ===
def obtener_mejor_partido(api_key):
    url = 'https://api.the-odds-api.com/v4/sports/soccer/odds/'
    params = {
        'apiKey': api_key,
        'regions': 'eu',
        'markets': 'h2h',
        'oddsFormat': 'decimal',
        'bookmakers': 'bet365,unibet,bwin'
    }
    r = requests.get(url, params=params)
    data = r.json()
    mejor = None
    mejor_score = -float('inf')

    for p in data:
        cuotas = []
        for casa in p['bookmakers']:
            for o in casa['markets'][0]['outcomes']:
                cuotas.append(o['price'])
        if len(cuotas) == 3:
            suma_prob = sum(1 / c for c in cuotas)
            score = 1 - suma_prob
            if score > mejor_score:
                mejor_score = score
                mejor = p

    if mejor:
        home = mejor['home_team']
        away = mejor['away_team']
        return home, away
    else:
        return None, None

# === 2. Buscar ID del equipo en API-FOOTBALL ===
def buscar_team_id(nombre_equipo, api_key):
    url = f"https://v3.football.api-sports.io/teams?search={nombre_equipo}"
    headers = {'x-apisports-key': api_key}
    r = requests.get(url, headers=headers)
    res = r.json()
    if res['response']:
        return res['response'][0]['team']['id']
    return None

# === 3. Modelo de regresión por equipo ===
def entrenar_modelo_equipo(team_id, league_id, season, api_key):
    url = f'https://v3.football.api-sports.io/fixtures?team={team_id}&league={league_id}&season={season}&status=FT'
    headers = {'x-apisports-key': api_key}
    partidos = requests.get(url, headers=headers).json()['response']

    data = []
    for p in partidos:
        goles = p['teams']['home']['goals'] if p['teams']['home']['id'] == team_id else p['teams']['away']['goals']
        localia = 1 if p['teams']['home']['id'] == team_id else 0
        fixture_id = p['fixture']['id']

        url_lineup = f'https://v3.football.api-sports.io/fixtures/lineups?fixture={fixture_id}'
        resp = requests.get(url_lineup, headers=headers).json()
        try:
            titulares = resp['response'][0]['startXI']
            ausencias = 0 if len(titulares) >= 11 else 1
        except:
            ausencias = 0

        data.append({'Goles': goles, 'Localia': localia, 'Ausencias': ausencias})

    df = pd.DataFrame(data)
    if df.empty or len(df) < 3:
        return None

    X = df[['Localia', 'Ausencias']]
    y = df['Goles']
    modelo = LinearRegression()
    modelo.fit(X, y)
    pred = modelo.predict([[1, 0]])[0]

    return {
        'coef_localia': modelo.coef_[0],
        'coef_ausencias': modelo.coef_[1],
        'intercepto': modelo.intercept_,
        'prediccion': pred
    }

# === 4. Ejecutar todo ===
def main():
    print("🔎 Buscando el mejor partido según las cuotas...")
    home_name, away_name = obtener_mejor_partido(ODDS_API_KEY)
    if not home_name:
        print("No se encontró un partido óptimo.")
        return

    print(f"\n🏟️ Partido sugerido: {home_name} vs {away_name}")

    home_id = buscar_team_id(home_name, API_FOOTBALL_KEY)
    away_id = buscar_team_id(away_name, API_FOOTBALL_KEY)
    if not home_id or not away_id:
        print("❌ No se pudo obtener el ID de alguno de los equipos.")
        return

    print("\n📈 Entrenando modelo para el equipo local...")
    m_home = entrenar_modelo_equipo(home_id, LEAGUE_ID, SEASON, API_FOOTBALL_KEY)
    print("📈 Entrenando modelo para el equipo visitante...")
    m_away = entrenar_modelo_equipo(away_id, LEAGUE_ID, SEASON, API_FOOTBALL_KEY)

    if not m_home or not m_away:
        print("❌ No se pudo entrenar un modelo para alguno de los equipos.")
        return

    print(f"\n🔴 {home_name}")
    print(f"  Localía: {m_home['coef_localia']:.2f}, Ausencias: {m_home['coef_ausencias']:.2f}, Intercepto: {m_home['intercepto']:.2f}")
    print(f"  🔮 Predicción de goles: {m_home['prediccion']:.2f}")

    print(f"\n🔵 {away_name}")
    print(f"  Localía: {m_away['coef_localia']:.2f}, Ausencias: {m_away['coef_ausencias']:.2f}, Intercepto: {m_away['intercepto']:.2f}")
    print(f"  🔮 Predicción de goles: {m_away['prediccion']:.2f}")

    if m_home['prediccion'] > m_away['prediccion']:
        print(f"\n✅ Pronóstico: gana {home_name}")
    elif m_home['prediccion'] < m_away['prediccion']:
        print(f"\n✅ Pronóstico: gana {away_name}")
    else:
        print("\n🤝 Pronóstico: empate")

if __name__ == "__main__":
    main()
