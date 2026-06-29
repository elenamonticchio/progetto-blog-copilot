"""
DOMAIN_TOPICS: l'universo dei topic che il blog vuole coprire.
Serve alla gap analysis del KG (get_coverage_gaps) per capire QUALI argomenti
del dominio non sono ancora stati trattati. Personalizzalo liberamente: più è
ricco e specifico, più i suggerimenti del planner saranno mirati e vari.
"""

DOMAIN_TOPICS = [
    # --- Generi ---
    "fantascienza", "horror", "thriller", "fantasy", "commedia",
    "documentari", "animazione", "true crime",
    # --- Registi / autori ---
    "Christopher Nolan", "Stanley Kubrick", "Greta Gerwig",
    "Steven Spielberg", "Martin Scorsese",
    # --- Piattaforme ---
    "Netflix", "Prime Video", "Apple TV+", "Disney+", "Max",
    # --- Formati / temi ricorrenti ---
    "miniserie", "adattamenti da libri", "reboot e sequel",
    "festival di cinema", "colonne sonore", "effetti visivi",
]
