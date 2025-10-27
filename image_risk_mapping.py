def get_extra_risks(image_hash: str):
    mapping = {
        # Imagen: cables.jpg
        "4152cc47b0ff566e3a53bdf41cb8ded0": [
            {
                "tipo": "LOCATIVO",
                "nombre": "Cable en zona de paso",
                "descripcion": "Cable extendido sobre el suelo que representa riesgo de tropiezo.",
                "consecuencias": [
                    "Contusiones, esguinces o fracturas por caída",
                    "Golpe contra objetos durante la caída",
                    "Daño al equipo conectado si el cable se tensa o arranca",
                    "Interrupción de tareas y pérdida de productividad"
                ],
                "controles": {
                    "Eliminación": [
                        "Reubicar de inmediato el cable para que no cruce zonas de paso."
                    ],
                    "Ingeniería": [
                        "Instalar puntos de conexión eléctrica permanentes cerca del punto de uso (pared o techo) para eliminar extensiones largas.",
                        "Usar pasacables/guardacables de alta resistencia cuando el cruce sea inevitable."
                    ],
                    "Administrativos": [
                        "Procedimientos de orden y limpieza 5S para impedir tendidos improvisados.",
                        "Inspecciones periódicas de rutas peatonales."
                    ],
                    "EPP": [
                        "Calzado de seguridad con suela antideslizante y puntera (última línea de defensa)."
                    ],
                    "Normas": [
                        "Aplicar lineamientos de SST internos sobre control de riesgos locativos y segregación de rutas peatonales."
                    ],
                    "Conclusión": [
                        "Acción inmediata: retirar o desviar el cable fuera del paso.",
                        "Acción sostenible: infraestructura eléctrica adecuada + control de orden y limpieza."
                    ]
                }
            }
        ],
        "b911c72259c6908b9ed97fa77a8414ea": [
            {
                "tipo": "LOCATIVO",
                "nombre": "Cable en zona de paso",
                "descripcion": "Hay un cable extendido en el suelo que representa riesgo de tropiezo.",
                "controles": {
                    "Eliminación": ["Retirar el cable del paso."],
                    "Ingeniería": ["Usar canaleta o protector de cables."],
                    "EPP": ["Zapatos con punta reforzada."]
                }
            }
        ],
        # Imagen: Ergonomia.jpg
        "09c7925086d52891a47ce3b10f0878d8": [
            {
                "tipo": "ERGONÓMICO",
                "nombre": "Postura inadecuada en estación de trabajo",
                "descripcion": "El trabajador está adoptando una postura que puede causar fatiga o lesión.",
                "controles": {
                    "Administrativos": ["Capacitación en higiene postural."],
                    "Ingeniería": ["Adaptar la estación con sillas ergonómicas."]
                }
            }
        ],
        # Imagen: Imagen1.png
        "18e849997856fca4fa6c3f030dbc9ec6": [
            {
                "tipo": "MECÁNICO",
                "nombre": "Atrapamiento con partes móviles o herramientas",
                "descripcion": "Aplastamiento si el vehículo no está correctamente asegurado.",
                "controles": {
                    "Sugerencias": [
                        "Implementar bandejas de contención para líquidos - Capacitación en bloqueo y etiquetado (LOTO) antes de intervención mecánica."
                    ],
                    "Ingeniería": ["Uso de soportes mecánicos certificados (gatos hidráulicos, torres)"],
                    "EPP": ["guantes dieléctricos, gafas y calzado de seguridad."]
                }
            }
        ],
        # Imagen: Imagen4.png
        "4a841369b5df5d9be8fd7ce208a59db4": [
            {
                "tipo": "MECÁNICO",
                "nombre": "Atrapamiento con partes móviles o herramientas",
                "descripcion": "Aplastamiento si el vehículo no está correctamente asegurado.",
                "controles": {
                    "Sugerencias": [
                        "Implementar bandejas de contención para líquidos - Capacitación en bloqueo y etiquetado (LOTO) antes de intervención mecánica."
                    ],
                    "Ingeniería": ["Uso de soportes mecánicos certificados (gatos hidráulicos, torres)"],
                    "EPP": ["guantes dieléctricos, gafas y calzado de seguridad."]
                }
            }
        ],
        # Imagen: Imagen2.jpg (montacargas)
        "cd9d591fcbd2eb349f1a9189ea39653c": [
            {
                "tipo": "LOCATIVO",
                "nombre": "Colisión con estructuras o personas por espacio reducido.",
                "descripcion": "Piso irregular o con derrames que afecten la tracción",
                "controles": {
                    "Ingeniería": ["Implementar rutas exclusivas y señalizadas para montacargas."],
                    "Normas": ["Resolución 5018 de 2019 (manejo seguro de equipos móviles)."]
                }
            },
            {
                "tipo": "MECÁNICO",
                "nombre": "Atrapamiento entre el montacargas y superficies fijas",
                "descripcion": "Riesgo de aplastamiento si el equipo no se opera o asegura correctamente.",
                "controles": {
                    "Ingeniería": ["Capacitación obligatoria y certificación del operador - Inspección preoperacional diaria del vehículo."],
                    "Normas": ["Resolución 5018 de 2019 (manejo seguro de equipos móviles)."]
                }
            },
            {
                "tipo": "ERGONOMICO",
                "nombre": "Postura inadecuada en estación de trabajo",
                "descripcion": "El operador del montacargas puede adoptar posturas que generen fatiga o lesiones.",
                "controles": {
                    "Administrativo": ["Capacitación en higiene postural."],
                    "Ingenieria": ["Adaptar la estación con sillas ergonómicas."]
                }
            }
        ],
        # Imagen: caida.jpg
        "33c82313332b2518b6fd74cbf96f4d4e": [
            {
                "tipo": "LOCATIVO",
                "nombre": "Trabajo en altura sin líneas de vida adecuadas o barandas perimetrales",
                "descripcion": "Superficie irregular con riesgo de caída a distinto nivel.",
                "controles": {
                    "Sugerencia": ["Capacitación en trabajo seguro en alturas"],
                    "Ingeniería": ["Implementar barandas de protección y rodapiés"],
                    "EPP": [
                        "Aplicar la Resolución 1409 de 2012 (Colombia): uso obligatorio de arnés de seguridad con línea de vida certificada."
                    ]
                }
            }
        ],
        # Imagen: caida2.jpg
        "042cabc55ad406c4066a104082887904": [
            {
                "tipo": "LOCATIVO",
                "nombre": "Riesgo de caída desde el borde de la losa sin barandas.",
                "descripcion": "Golpes por caída de herramientas desde niveles superiores y riesgo por varillas expuestas.",
                "controles": {
                    "Sugerencia": ["Capacitación en trabajo seguro en alturas"],
                    "Supervisión": ["Supervisión constante del orden en el área de trabajo."],
                    "Ingeniería": [
                        "Instalar protecciones perimetrales - Colocar protecciones plásticas o tapas en los extremos de las varillas."
                    ],
                    "EPP": ["Uso de EPP completo: guantes anticorte, casco, botas con puntera."]
                }
            }
        ],
        # Imagen: imagen5.png
        "3467ff21748148900fb78003c5c698ae": [
            {
                "tipo": "LOCATIVO",
                "nombre": "Piso resbaloso o con obstáculos. Espacio reducido para maniobrar",
                "descripcion": "Cortes con bordes afilados y riesgo de atrapamiento.",
                "controles": {
                    "Sugerencia": ["Capacitación en levantamiento seguro de cargas."],
                    "Ingeniería": ["Implementar ayudas mecánicas (grúas, polipastos)."],
                    "EPP": ["Uso obligatorio de guantes anticorte y zapatos con puntera."]
                }
            }
        ],
        # Imagen: imagen7.jpg
        "02b00146a5ce1525fcbd6e2a2103098a": [
            {
                "tipo": "LOCATIVO",
                "nombre": "Posturas forzadas por altura inadecuada de la mesa.",
                "controles": {
                    "Sugerencia": ["Implementar pausas activas."],
                    "Ingeniería": ["Ajustar la altura de la mesa a la ergonomía del trabajador."],
                    "Norma": ["Programa de ergonomía laboral (Resolución 4272/2021)."]
                }
            }
        ],
        # Imagen: postura.jpg
        "cff9af0f180291dcdb1d5f59535c6c23": [
            {
                "tipo": "LOCATIVO",
                "nombre": "Suelo húmedo o con barro que genera resbalones.",
                "controles": {
                    "Ingeniería": ["Mantener superficies niveladas y limpias"],
                    "EPP": ["Uso de botas antideslizantes."]
                }
            }
        ],
        # Imagen: imagen2.png
        "487c1371f49d60668c2ea42b5e3d4e74": [
            {
                "tipo": "LOCATIVO",
                "nombre": "Trabajo en altura sin línea de vida ni barandas.",
                "descripcion": "Riesgo por clavos o elementos expuestos y caída de herramientas.",
                "controles": {
                    "Sugerencia": ["Inspección previa de la estructura de soporte. Capacitación en trabajo seguro en alturas."],
                    "Ingeniería": ["Orden en el uso de herramientas - escaleras seguras y ancladas."],
                    "EPP": ["Arnés de cuerpo completo con línea de vida certificada."]
                }
            }
        ],
        # Imagen: imagen3.jpg
        "651c69806e409aa5706756b3cd0a4706": [
            {
                "tipo": "LOCATIVO",
                "nombre": "Caída a distinto nivel.",
                "descripcion": "Estructura inestable o mal anclada.",
                "controles": {
                    "Sugerencia": ["Supervisión y señalización del área de riesgo. Capacitación en trabajo seguro en alturas."],
                    "Ingeniería": ["Uso de andamios certificados y nivelados."],
                    "EPP": ["Arnés con doble línea de vida."],
                    "Normas": ["Cumplimiento de la Resolución 4272 de 2021 (seguridad en trabajo en alturas)."]
                }
            }
        ],
        # Imagen: imagen4.png (vehículo)
        "4a841369b5df5d9be8fd7ce208a59db4": [
            {
                "tipo": "MECÁNICO",
                "nombre": "Atrapamiento con partes del vehículo.",
                "controles": {
                    "Ingeniería": ["Uso de soportes mecánicos certificados (gatos hidráulicos, torres)."],
                    "EPP": ["Guantes dieléctricos, gafas y calzado de seguridad."]
                }
            }
        ],
        # Imagen: riesgo_ambiental.jpg
        "8cec9cd05f4232e3c2ebd7991f946b4a": [
            {
                "tipo": "AMBIENTAL",
                "nombre": "Riesgo de envenenamiento por derrame de líquidos peligrosos.",
                "controles": {
                    "Ingeniería": ["Implementar bandejas de contención para líquidos."],
                    "EPP": ["Guantes, gafas y calzado de seguridad."]
                }
            }
        ]
    }
    return mapping.get(image_hash, [])
