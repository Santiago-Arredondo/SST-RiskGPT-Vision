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
                        "RETIE: seguridad de instalaciones y elementos eléctricos; calidad y protección de extensiones.",
                "NTC 2050 (Código Eléctrico Colombiano): calibre de conductores, protección contra sobrecorriente y puesta a tierra.",
                "Resolución 5018 de 2019: control de peligros en el uso de energía eléctrica e inspecciones preoperacionales.",
                "Decreto 1072 de 2015 y Resolución 0312 de 2019: SG-SST, identificación y control de riesgos locativos, mantenimiento y orden y limpieza."
           
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
    "descripcion": "El trabajador adopta una postura forzada o estática prolongada debido a la inadecuada disposición del mobiliario y la configuración del puesto, generando riesgo de trastornos musculoesqueléticos y fatiga visual.",
    "consecuencias": [
        "Dolor cervical, lumbar y dorsal por mala alineación corporal.",
        "Síndrome del túnel carpiano y tendinitis por uso prolongado del teclado o ratón en posición inadecuada.",
        "Fatiga visual y cefaleas por altura o brillo incorrecto del monitor.",
        "Desestabilización o sobreesfuerzo lumbar por maleta u objetos colgados en la silla.",
        "Tropiezos o daños a equipos por cableado mal dispuesto bajo el escritorio."
    ],
    "controles": {
        "Eliminación": [
            "Eliminar o reorganizar elementos innecesarios del área de trabajo que limiten la movilidad o el ajuste ergonómico.",
            "Retirar objetos colgados del espaldar de la silla (maletas, botellas, etc.) y asignar espacios de almacenamiento adecuados."
        ],
        "Ingeniería": [
            "Instalar sillas ergonómicas certificadas con ajuste de altura, espaldar reclinable y soporte lumbar.",
            "Usar soportes para laptop o elevadores que coloquen la parte superior de la pantalla a la altura de los ojos.",
            "Implementar teclado y ratón externos ergonómicos que permitan mantener antebrazos paralelos al suelo y muñecas rectas.",
            "Ubicar el monitor a una distancia de 50–70 cm y con el centro ligeramente por debajo de la línea de visión.",
            "Asignar reposapiés ajustable cuando la altura del escritorio impida que los pies se apoyen completamente en el suelo.",
            "Canalizar o asegurar el cableado para evitar tropiezos o tirones accidentales de equipos."
        ],
        "Administrativos": [
            "Capacitación en higiene postural y ajuste del mobiliario según las medidas antropométricas del trabajador.",
            "Implementar pausas activas obligatorias al menos dos veces por jornada laboral.",
            "Establecer rotación de tareas para reducir posturas mantenidas y esfuerzos repetitivos.",
            "Realizar inspecciones ergonómicas periódicas en los puestos de trabajo.",
            "Registrar y corregir observaciones derivadas de las evaluaciones ergonómicas y del SG-SST."
        ],
        "Señalización": [
            "Colocar infografías o recordatorios visibles sobre posturas correctas y pausas activas en zonas de trabajo.",
            "Identificar puestos que ya fueron ajustados ergonómicamente para seguimiento por parte del SG-SST."
        ],
        "EPP": [
            "No aplica directamente. En entornos de oficina, la prevención se logra mediante ingeniería y control administrativo, no con EPP."
        ],
        "Normas": [
            "Decreto 1072 de 2015 (Decreto Único Reglamentario del Sector Trabajo): Artículo 2.2.4.6.15, exige la identificación de peligros ergonómicos y medidas de control.",
            "Resolución 0312 de 2019 (Estándares Mínimos del SG-SST): Estándar 6, prevención de Trastornos Musculoesqueléticos (TME).",
            "Decreto 1477 de 2014 (Tabla de Enfermedades Laborales): reconoce TME como enfermedades de origen laboral.",
            "Guía Técnica Colombiana GTC 45: clasificación y valoración del peligro ergonómico según frecuencia, exposición y consecuencias.",
            "Guías del Ministerio del Trabajo y de las ARL: orientaciones para la evaluación ergonómica de puestos de oficina."
        ],
        "Conclusión": [
            "El ajuste ergonómico del mobiliario y la educación postural son esenciales para prevenir TME.",
            "El empleador debe garantizar el cumplimiento del SG-SST, capacitar y evaluar periódicamente las condiciones ergonómicas del puesto.",
            "Un entorno ergonómicamente adecuado mejora la productividad y reduce el ausentismo laboral."
        ]
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
[
    {
        "tipo": "LOCATIVO",
        "nombre": "Colisión con estructuras o personas por espacio reducido",
        "descripcion": "Riesgo de atropellamiento o colisión con estructuras debido a pasillos estrechos, visibilidad limitada y falta de segregación entre peatones y vehículos.",
        "consecuencias": [
            "Atropellamiento o aplastamiento de peatones en zonas de paso compartido.",
            "Colisión con estanterías, columnas o muros, generando daños estructurales o desplome de cargas.",
            "Vuelco del equipo por giros bruscos, exceso de velocidad o piso irregular.",
            "Caída de la carga por mala estiba o inclinación incorrecta de las horquillas."
        ],
        "controles": {
            "Eliminación": [
                "Evitar el tránsito de peatones por áreas designadas para montacargas mediante separación física o control de acceso."
            ],
            "Ingeniería": [
                "Implementar rutas exclusivas, demarcadas y señalizadas para montacargas y peatones.",
                "Instalar espejos convexos en esquinas ciegas y alarmas luminosas tipo ‘Blue Light’.",
                "Mantener el piso nivelado, libre de obstáculos y con textura antideslizante.",
                "Colocar protecciones de impacto en columnas, estanterías y muros expuestos."
            ],
            "Administrativos": [
                "Establecer límites de velocidad (máx. 10 km/h) y señalización de prioridad de paso.",
                "Definir zonas de carga y descarga con visibilidad y espacio adecuados.",
                "Supervisión activa del cumplimiento de rutas y normas internas de tránsito.",
                "Capacitar al personal en convivencia vial interna y seguridad industrial."
            ],
            "Señalización": [
                "Demarcar con pintura y señal vertical las rutas vehiculares y peatonales.",
                "Colocar letreros de advertencia de montacargas en movimiento."
            ],
            "EPP": [
                "Chaleco reflectivo de alta visibilidad.",
                "Calzado de seguridad con suela antideslizante y puntera de protección."
            ],
            "Normas": [
                "Resolución 5018 de 2019 (Manejo seguro de equipos móviles).",
                "Decreto 1072 de 2015 (SG-SST: identificación y control de riesgos locativos).",
                "Código Nacional de Tránsito (Ley 769 de 2002): aplicación de principios de seguridad vial dentro de instalaciones.",
                "NTC 4848 (Seguridad en montacargas y carretillas elevadoras)."
            ],
            "Conclusión": [
                "Separar físicamente las rutas peatonales y vehiculares y garantizar visibilidad en giros e intersecciones es esencial para prevenir accidentes."
            ]
        }
    },
    {
        "tipo": "MECÁNICO",
        "nombre": "Atrapamiento entre el montacargas y superficies fijas",
        "descripcion": "Riesgo de aplastamiento del operador o de otros trabajadores debido al manejo inadecuado del equipo, fallas mecánicas o deficiente mantenimiento preventivo.",
        "consecuencias": [
            "Aplastamiento de extremidades o torso por maniobras inseguras.",
            "Golpes y fracturas por pérdida de control o vuelco del montacargas.",
            "Incendio o explosión por fugas del sistema de gas (LPG).",
            "Exposición a gases de combustión (monóxido de carbono) por ventilación deficiente."
        ],
        "controles": {
            "Eliminación": [
                "Prohibir el uso del montacargas por personal no certificado.",
                "Evitar la operación del equipo en zonas con inclinaciones, desniveles o con superficie inestable."
            ],
            "Ingeniería": [
                "Equipar el montacargas con cinturón de seguridad, alarma de reversa sonora y luminosa.",
                "Implementar mantenimiento preventivo documentado en frenos, dirección, hidráulica y sistema de gas.",
                "Verificar el estado del cilindro de gas propano (válvula, manguera y anclaje).",
                "Mantener ventilación adecuada en zonas de operación cerradas."
            ],
            "Administrativos": [
                "Certificación obligatoria del operador y licencia de conducción vigente.",
                "Checklist preoperacional diario (frenos, luces, bocina, dirección, fluidos, cilindro de gas).",
                "Procedimiento de trabajo seguro para el manejo de gas LPG y cambio de cilindros.",
                "Supervisión del cumplimiento de normas internas de operación y tránsito.",
                "Plan de emergencia específico ante incendios por gas o colisiones."
            ],
            "Señalización": [
                "Carteles visibles de velocidad máxima, zonas de carga y precaución con montacargas.",
                "Avisos de uso obligatorio de cinturón de seguridad y chaleco reflectivo."
            ],
            "EPP": [
                "Chaleco reflectivo, casco de seguridad, gafas protectoras y botas con puntera de acero.",
                "Protección auditiva (si se superan los 85 dB por ruido del motor)."
            ],
            "Normas": [
                "Resolución 5018 de 2019 (manejo seguro de equipos móviles).",
                "Decreto 1072 de 2015 y Resolución 0312 de 2019 (SG-SST: gestión de equipos críticos).",
                "NTC 4848 (Seguridad en montacargas y carretillas elevadoras)."
            ],
            "Conclusión": [
                "La inspección preoperacional, la certificación del operador y la segregación de áreas son medidas clave para evitar accidentes por atrapamiento o vuelco."
            ]
        }
    },
    {
        "tipo": "ERGONÓMICO",
        "nombre": "Postura inadecuada del operador de montacargas",
        "descripcion": "El operador mantiene una postura estática prolongada con exposición a vibración y ruido, lo que genera fatiga muscular y trastornos musculoesqueléticos.",
        "consecuencias": [
            "Dolor lumbar y cervical por postura estática prolongada.",
            "Trastornos musculoesqueléticos (TME) por vibración y movimientos repetitivos.",
            "Hipoacusia o pérdida auditiva inducida por ruido del motor.",
            "Fatiga general y disminución del nivel de atención durante la jornada."
        ],
        "controles": {
            "Eliminación": [
                "Rotar tareas entre diferentes operarios para reducir la exposición prolongada al mismo tipo de esfuerzo."
            ],
            "Ingeniería": [
                "Cabinas con asientos ergonómicos ajustables y suspensión para amortiguar la vibración.",
                "Mejorar aislamiento acústico del motor y mantenimiento de sistemas de escape.",
                "Instalar plataformas niveladas y rampas adecuadas para evitar inclinaciones prolongadas del cuerpo."
            ],
            "Administrativos": [
                "Capacitación en higiene postural y pausas activas específicas para operadores de montacargas.",
                "Programa de vigilancia epidemiológica para control de TME y ruido ocupacional.",
                "Evaluación médica ocupacional periódica (agudeza visual, auditiva y osteomuscular).",
                "Supervisión de cumplimiento de pausas, estiramientos y cambio de posturas durante la jornada."
            ],
            "EPP": [
                "Protección auditiva adecuada (tapones o copas).",
                "Calzado ergonómico con absorción de impactos.",
                "Ropa de trabajo cómoda y reflectiva para movilidad segura."
            ],
            "Normas": [
                "Decreto 1072 de 2015 (SG-SST: control de riesgos ergonómicos y físicos).",
                "Resolución 0312 de 2019 (Estándares mínimos de seguridad y salud en el trabajo).",
                "NTC 4848 y guías del Ministerio del Trabajo sobre ergonomía en equipos de manejo de materiales."
            ],
            "Conclusión": [
                "El confort del operador y la reducción de vibración son esenciales para prevenir TME y fatiga.",
                "El control ergonómico complementa la seguridad mecánica para lograr una operación segura y sostenible."
            ]
        }
    }
]
        ],
        # Imagen: caida.jpg
        "33c82313332b2518b6fd74cbf96f4d4e": [
            {
    "tipo": "LOCATIVO",
    "nombre": "Trabajo en altura sin líneas de vida adecuadas o barandas perimetrales",
    "descripcion": "Superficie irregular con riesgo de caída a distinto nivel por ausencia de protecciones colectivas (barandas, rodapiés, redes) o líneas de vida certificadas.",
    "consecuencias": [
        "Caída a distinto nivel con lesiones graves o mortales.",
        "Golpe contra estructuras o equipos en niveles inferiores.",
        "Caída de herramientas o materiales sobre otros trabajadores.",
        "Daños a infraestructura o equipos por impacto.",
        "Interrupción de labores y paros por incidentes graves."
    ],
    "controles": {
        "Eliminación": [
            "Evitar realizar trabajos en altura siempre que la tarea pueda ejecutarse desde nivel seguro.",
            "Reubicar tareas o utilizar herramientas con mango extensible desde el suelo cuando sea posible."
        ],
        "Ingeniería": [
            "Instalar barandas perimetrales con rodapié, redes de seguridad o líneas de vida horizontales/verticales certificadas.",
            "Usar andamios o plataformas de trabajo niveladas y con sistema de acceso seguro.",
            "Verificar resistencia estructural del punto de anclaje antes del uso de líneas de vida.",
            "Asegurar superficies de trabajo limpias y sin objetos que puedan causar tropiezos o resbalones."
        ],
        "Administrativos": [
            "Implementar el Programa de Protección Contra Caídas según la Resolución 1409 de 2012.",
            "Exigir que solo personal certificado en trabajo seguro en alturas realice las tareas.",
            "Realizar inspecciones previas y durante la ejecución para verificar el uso del sistema de protección.",
            "Establecer permisos de trabajo en altura con supervisión continua.",
            "Capacitar en rescate y primeros auxilios en altura."
        ],
        "Señalización": [
            "Delimitar y señalizar el área inferior como zona de exclusión durante la actividad.",
            "Colocar avisos de advertencia visibles sobre el riesgo de caída."
        ],
        "EPP": [
            "Arnés de cuerpo completo con línea de vida certificada y doble eslinga con absorbedor de energía.",
            "Casco con barbuquejo, guantes antideslizantes y calzado de seguridad con suela antideslizante.",
            "Gafas de protección cuando exista riesgo de caída de partículas."
        ],
        "Normas": [
            "Resolución 1409 de 2012 (Trabajo Seguro en Alturas).",
            "Resolución 4272 de 2021 (Requisitos de protección contra caídas).",
            "Decreto 1072 de 2015 y Resolución 0312 de 2019 (SG-SST: gestión del riesgo locativo).",
            "NTC 1641 y NTC 1642 (Requisitos técnicos para equipos de protección en alturas)."
        ],
        "Conclusión": [
            "Antes de realizar trabajos en altura, deben implementarse controles colectivos (barandas, redes o líneas de vida).",
            "Si no existen protecciones adecuadas, la labor debe suspenderse hasta asegurar condiciones seguras.",
            "La formación, inspección constante y el cumplimiento estricto de la normatividad son esenciales para prevenir caídas fatales."
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
    "descripcion": "El trabajador manipula láminas metálicas en un área estrecha y con piso irregular o contaminado con residuos, lo que incrementa el riesgo de cortes, atrapamientos y caídas.",
    "consecuencias": [
        "Atrapamiento o amputación por contacto con el punto de operación de una máquina de prensado o corte.",
        "Cortes profundos o punciones por bordes filosos de las láminas metálicas.",
        "Caída del material sobre pies o manos, generando fracturas o contusiones.",
        "Lesiones musculoesqueléticas (TME) por manipulación manual de cargas pesadas y posturas forzadas.",
        "Golpes o caídas por resbalones causados por aceites, virutas o piezas en el piso."
    ],
    "controles": {
        "Eliminación": [
            "Evitar la manipulación manual de láminas pesadas mediante la automatización del proceso.",
            "Eliminar obstáculos y mantener libre de residuos el espacio de trabajo para garantizar movilidad segura."
        ],
        "Ingeniería": [
            "Implementar ayudas mecánicas como grúas, polipastos o brazos articulados para el levantamiento de láminas metálicas.",
            "Instalar guardas físicas o cercos de seguridad que impidan el acceso al punto de operación de la máquina durante su ciclo.",
            "Incorporar dispositivos de detección (cortinas de luz o sensores) que detengan el movimiento de la máquina al detectar la presencia del operario.",
            "Adoptar controles bimanuales para impedir la activación de la máquina si las manos están dentro del área de riesgo.",
            "Instalar pisos antideslizantes y sistemas de drenaje que eviten la acumulación de aceites o residuos metálicos.",
            "Disponer de iluminación suficiente para visibilidad adecuada en el área de trabajo."
        ],
        "Administrativos": [
            "Establecer un Procedimiento de Trabajo Seguro (PTS) para la manipulación de láminas metálicas y operación de máquinas de prensado o corte.",
            "Capacitar al personal en técnicas seguras de levantamiento manual y uso de ayudas mecánicas, conforme a las recomendaciones ergonómicas.",
            "Implementar un programa de inspecciones diarias de guardas, sensores, cortinas de luz y bloqueos (LOTO).",
            "Exigir el uso de dispositivos de bloqueo y etiquetado (LOTO) durante tareas de mantenimiento o ajuste de máquinas.",
            "Aplicar el programa de orden y limpieza (5S) para mantener los pisos secos, sin virutas ni residuos.",
            "Supervisar el cumplimiento del procedimiento de levantamiento seguro y la verificación previa de dispositivos de seguridad."
        ],
        "Señalización": [
            "Marcar zonas de riesgo con líneas amarillas y avisos de 'Peligro: partes móviles' y 'Uso obligatorio de EPP'.",
            "Demarcar rutas seguras de tránsito y áreas de maniobra para prevenir interferencias entre trabajadores y equipos.",
            "Colocar señalización visible sobre el límite de carga y precauciones de atrapamiento."
        ],
        "EPP": [
            "Guantes anticorte de kevlar o cuero reforzado para manipulación de metal.",
            "Calzado de seguridad con puntera de acero y suela antideslizante.",
            "Casco de seguridad obligatorio en áreas con riesgo de caída de materiales.",
            "Gafas de seguridad para proteger de astillas o partículas desprendidas.",
            "Ropa de trabajo ajustada y resistente a cortes o enganches."
        ],
        "Normas": [
            "Decreto 1072 de 2015: exige control de peligros mecánicos, locativos y biomecánicos dentro del SG-SST.",
            "Resolución 0312 de 2019: establece la obligación de mantenimiento de equipos, bloqueo/etiquetado (LOTO) y suministro de EPP.",
            "NTC 4991 y NTC 5613: lineamientos técnicos para guardas de seguridad y dispositivos de enclavamiento en máquinas.",
            "Decreto 1072 de 2015 - Riesgo Biomecánico: control mediante rediseño de tareas o ayudas mecánicas.",
            "Resolución 773 de 2021: exige el uso de Fichas de Datos de Seguridad (SDS/FDS) para fluidos de corte o refrigerantes.",
            "Código Sustantivo del Trabajo, Art. 348: obligación del empleador de suministrar máquinas y herramientas seguras."
        ],
        "Conclusión": [
            "El riesgo de atrapamiento y corte en procesos de prensado y manipulación de láminas metálicas es alto y requiere control simultáneo de ingeniería y capacitación.",
            "El uso de guardas, sensores, ayudas mecánicas y EPP adecuados previene lesiones graves y asegura el cumplimiento del SG-SST.",
            "El orden, la limpieza y la automatización son esenciales para eliminar riesgos de resbalón y atrapamiento en entornos industriales."
        ]
    }
}

        ],
        # Imagen: imagen7.jpg
        "02b00146a5ce1525fcbd6e2a2103098a": [
            {
    "tipo": "LOCATIVO",
    "nombre": "Posturas forzadas por altura inadecuada de la mesa",
    "descripcion": "El trabajador realiza tareas sobre una mesa que no se ajusta a su altura, obligándolo a inclinar el tronco y adoptar posturas estáticas o forzadas, lo que genera fatiga, dolor muscular y riesgo de lesiones osteomusculares.",
    "consecuencias": [
        "Trastornos musculoesqueléticos (TME) en cuello, espalda y hombros por posturas mantenidas.",
        "Fatiga visual y cefaleas por iluminación deficiente o esfuerzo visual prolongado.",
        "Caídas o tropiezos por desorden o cables en el entorno de trabajo.",
        "Cortes o abrasiones por contacto con herramientas o materiales mal ubicados.",
        "Aumento de la fatiga general y disminución del rendimiento laboral."
    ],
    "controles": {
        "Eliminación": [
            "Eliminar la necesidad de inclinarse o permanecer en posturas forzadas mediante la adecuación del puesto.",
            "Reorganizar el área de trabajo para mantener despejadas las zonas de tránsito y evitar cables o materiales en el suelo."
        ],
        "Ingeniería": [
            "Instalar mesas o bancos de trabajo con altura regulable para adaptarse a la estatura del trabajador y tipo de tarea (de pie o sentado).",
            "Implementar superficies inclinables o móviles que acerquen el plano de trabajo al campo visual del trabajador.",
            "Asegurar que la mesa sea estable, sin bordes filosos ni rebabas, y anclada al suelo o estructura para evitar desplazamientos.",
            "Proveer iluminación focalizada o de aumento (lámparas con lupa) para tareas de precisión y estudios de luxometría periódicos.",
            "Usar herramientas ligeras y con diseño ergonómico (mangos acolchados, sin cables tensos).",
            "Controlar el ruido y la vibración mediante mantenimiento de herramientas y uso de bases amortiguadas en las mesas."
        ],
        "Administrativos": [
            "Diseñar e implementar un Procedimiento de Higiene Postural y ergonomía en tareas de precisión o montaje.",
            "Establecer rotación de tareas entre actividades que exigen postura fija y tareas con movimiento.",
            "Aplicar un programa formal de pausas activas (cada 45–60 minutos) con ejercicios dirigidos de cuello, espalda y hombros.",
            "Capacitar a los trabajadores sobre posturas neutras, ajuste de mobiliario y señales de fatiga postural.",
            "Aplicar la metodología 5S para mantener el área de trabajo ordenada y limpia.",
            "Verificar la ergonomía del puesto mediante inspecciones del SG-SST y seguimiento en el Programa de Vigilancia Epidemiológica Osteomuscular (PVE)."
        ],
        "Señalización": [
            "Ubicar carteles con recordatorios de higiene postural y pausas activas.",
            "Identificar las áreas donde se requiere el uso de EPP y la inspección de herramientas eléctricas (LOTO).",
            "Señalizar cables, zonas de tránsito y áreas de manipulación de herramientas eléctricas para prevenir tropiezos."
        ],
        "EPP": [
            "Gafas de seguridad para evitar proyección de partículas o virutas (correctamente usadas en la imagen).",
            "Guantes de precisión o anticorte según la naturaleza del material manipulado.",
            "Calzado de seguridad con suela antideslizante y puntera de protección.",
            "Tapones o protectores auditivos si el nivel de ruido supera los 85 dB (8 horas)."
        ],
        "Normas": [
            "Resolución 0312 de 2019 (Estándares mínimos del SG-SST): identifica y controla peligros físicos y ergonómicos.",
            "Decreto 1072 de 2015 (SG-SST): obliga a implementar el Programa de Vigilancia Epidemiológica para riesgo osteomuscular.",
            "GTC 45: guía técnica para la identificación y valoración del riesgo biomecánico y físico (iluminación y ruido).",
            "Resolución 4272 de 2021: establece lineamientos para programas de ergonomía laboral y pausas activas.",
            "Código Sustantivo del Trabajo, Art. 348: exige condiciones seguras de mobiliario y herramientas de trabajo."
        ],
        "Conclusión": [
            "El riesgo crítico es la postura forzada derivada de una mesa de altura fija e inadecuada.",
            "La prioridad es rediseñar el puesto con mesas ajustables, mejorar la iluminación y capacitar en higiene postural.",
            "La prevención de TME y la ergonomía del puesto dependen de la combinación de ingeniería, pausas activas y orden del entorno."
        ]
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
    "nombre": "Caída a distinto nivel",
    "descripcion": "El trabajador realiza tareas en una estructura inestable o mal anclada, sin sistemas de protección contra caídas ni plataformas seguras de acceso, exponiéndose a una caída desde altura con consecuencias potencialmente fatales.",
    "consecuencias": [
        "Caída a distinto nivel con lesiones graves o muerte.",
        "Golpe contra estructuras, herramientas o el suelo tras la caída.",
        "Colapso parcial de la estructura de encofrado por sobrecarga o mal anclaje.",
        "Proyección de partículas o astillas al golpear elementos con herramientas manuales.",
        "Daños a la infraestructura o interrupción de labores por accidentes graves."
    ],
    "controles": {
        "Eliminación": [
            "Evitar realizar tareas en altura si pueden ejecutarse desde un nivel inferior o mediante herramientas extensibles.",
            "Prohibir el uso de estructuras de encofrado como plataforma de trabajo si no están diseñadas ni certificadas para ello."
        ],
        "Ingeniería": [
            "Instalar barandas perimetrales con rodapié o redes de seguridad (protecciones pasivas) en el área de trabajo.",
            "Diseñar e instalar puntos de anclaje fijos y certificados para conectar sistemas de protección contra caídas.",
            "Usar andamios certificados y plataformas modulares niveladas, diseñadas específicamente para montaje o desencofrado.",
            "Verificar la estabilidad estructural y el anclaje de los elementos antes de iniciar la tarea.",
            "Mantener el área libre de objetos o materiales sueltos que puedan causar tropiezos o desequilibrio."
        ],
        "Administrativos": [
            "Implementar un Procedimiento de Trabajo Seguro (PTS) para actividades de encofrado y desencofrado.",
            "Requerir permiso de trabajo en alturas (PTA) firmado por el coordinador competente antes de iniciar la tarea.",
            "Garantizar la presencia de un coordinador de alturas para la supervisión permanente de las actividades.",
            "Capacitar al personal en trabajo seguro en alturas (curso avanzado vigente) y técnicas de anclaje y desplazamiento.",
            "Aplicar el principio de doble aseguramiento (100% de anclaje) al utilizar sistemas de dos eslingas.",
            "Realizar inspecciones diarias a los sistemas de anclaje, andamios, líneas de vida y EPP antes del uso."
        ],
        "Señalización": [
            "Delimitar y señalizar las áreas de trabajo en altura como zonas de riesgo restringido.",
            "Instalar carteles de advertencia de peligro de caída y prohibición de paso a personal no autorizado."
        ],
        "EPP": [
            "Arnés de cuerpo completo certificado (NTC 2037) correctamente ajustado.",
            "Eslingas con absorbedor de energía conectadas a un punto de anclaje fijo o línea de vida.",
            "Casco de seguridad con barbuquejo (uso obligatorio en todo trabajo en altura).",
            "Gafas de seguridad contra proyección de partículas y polvo.",
            "Guantes de protección mecánica y botas con suela antideslizante y puntera de acero."
        ],
        "Normas": [
            "Resolución 4272 de 2021: requisitos técnicos y administrativos para trabajos en alturas (aplicación obligatoria desde 1.5 metros).",
            "Resolución 1409 de 2012: reglamenta los procedimientos de protección contra caídas en alturas.",
            "Decreto 1072 de 2015: gestión de peligros locativos dentro del SG-SST.",
            "Resolución 0312 de 2019: estándares mínimos del Sistema de Gestión de Seguridad y Salud en el Trabajo.",
            "NTC 2037: especificaciones técnicas para arneses de cuerpo completo y elementos de conexión."
        ],
        "Conclusión": [
            "La situación observada representa un riesgo inaceptable según la normativa colombiana por ausencia de sistemas visibles de protección contra caídas.",
            "Ninguna tarea debe realizarse en altura sin barandas, anclajes o arnés certificado, ni sin un permiso de trabajo autorizado.",
            "El cumplimiento estricto de la Resolución 4272 de 2021 es esencial para prevenir accidentes fatales en labores de construcción o mantenimiento en altura."
        ]
    }
}

        ],
        # Imagen: imagen4.png (vehículo)
        "4a841369b5df5d9be8fd7ce208a59db4": [
            {
    "tipo": "MECÁNICO",
    "nombre": "Atrapamiento con partes del vehículo",
    "descripcion": "El trabajador realiza mantenimiento o reparación de un vehículo elevado sin soportes mecánicos certificados, exponiéndose a riesgo de aplastamiento, contacto con fluidos peligrosos y contaminación ambiental por derrames.",
    "consecuencias": [
        "Aplastamiento por caída del vehículo debido a falla del sistema hidráulico o mal uso del gato.",
        "Golpes o atrapamiento con partes móviles del vehículo o herramientas durante la reparación.",
        "Irritación cutánea o dermatitis por contacto con aceites, líquidos de frenos o anticongelantes.",
        "Contaminación ambiental y riesgo de resbalón por derrames de fluidos en el piso.",
        "Inhalación de vapores de combustión o solventes en áreas sin ventilación adecuada."
    ],
    "controles": {
        "Eliminación": [
            "Prohibir el ingreso bajo vehículos sostenidos únicamente por gatos hidráulicos sin torres o soportes certificados.",
            "Evitar realizar tareas bajo vehículos si no existen condiciones seguras de anclaje y bloqueo."
        ],
        "Ingeniería": [
            "Usar soportes mecánicos certificados (torres, gatos de seguridad) con capacidad adecuada para el vehículo.",
            "Implementar bloqueos mecánicos redundantes después de la elevación hidráulica.",
            "Utilizar bandejas de contención o recipientes móviles para recoger aceites y fluidos sin derrames.",
            "Disponer de material absorbente (kit antiderrame, arena o aserrín) en el área de trabajo.",
            "Garantizar ventilación natural o forzada para eliminar vapores o gases de combustión.",
            "Revisar periódicamente los sistemas de elevación y los elementos estructurales de soporte."
        ],
        "Administrativos": [
            "Establecer un Procedimiento de Trabajo Seguro (PTS) para operaciones de elevación de vehículos.",
            "Instruir sobre el correcto posicionamiento de las garras o puntos de apoyo del elevador y activación de seguros mecánicos.",
            "Capacitar al personal en el manejo de fluidos peligrosos y en la gestión de residuos aceitosos (PGIRP).",
            "Implementar un programa de orden y limpieza (5S) para mantener pisos secos y despejados.",
            "Exigir la inspección preoperacional del equipo de elevación antes de cada uso.",
            "Documentar el mantenimiento preventivo del elevador, sistemas hidráulicos y tanques de gas (si aplica)."
        ],
        "Señalización": [
            "Demarcar zonas de mantenimiento con advertencias visibles: 'Vehículo en reparación' y 'Prohibido el paso'.",
            "Etiquetar áreas de almacenamiento de aceites y residuos peligrosos.",
            "Instalar carteles informativos sobre puntos de bloqueo y procedimientos de elevación segura."
        ],
        "EPP": [
            "Gafas de seguridad para evitar salpicaduras de aceites o partículas.",
            "Guantes de nitrilo o resistentes a químicos y grasas.",
            "Calzado de seguridad con puntera reforzada y suela antideslizante.",
            "Ropa de trabajo tipo overol que cubra la piel y minimice la exposición.",
            "Mascarilla o respirador cuando se trabaje con solventes o en espacios poco ventilados."
        ],
        "Normas": [
            "Decreto 1072 de 2015 (SG-SST): exige identificación y control de peligros mecánicos y químicos.",
            "Resolución 0312 de 2019 (Estándares mínimos del SG-SST): mantenimiento preventivo de equipos e inspecciones periódicas.",
            "Decreto 1076 de 2015 (Gestión ambiental): gestión integral de residuos peligrosos.",
            "Resolución 773 de 2021: adopta el Sistema Globalmente Armonizado (SGA) para la clasificación y rotulado de productos químicos.",
            "NTC 4831 y NTC 5801: lineamientos técnicos para elevadores hidráulicos y equipos de elevación.",
            "PGIRP (Plan de Gestión Integral de Residuos Peligrosos): obligatorio para manejo de aceites usados."
        ],
        "Conclusión": [
            "El uso de soportes certificados, la correcta gestión de residuos peligrosos y el mantenimiento del equipo de elevación son esenciales para prevenir aplastamientos e incidentes ambientales.",
            "No se debe trabajar bajo un vehículo sin bloqueos mecánicos activados ni EPP completo.",
            "El cumplimiento del SG-SST y las normas ambientales garantiza la seguridad del trabajador y del entorno."
        ]
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
        ],
    "0a710f4503124c05684a611e7b1c6d36": [
    {
        "tipo": "LOCATIVO",
        "nombre": "Superficie resbaladiza por derrame",
        "descripcion": "Líquidos o aceites en el piso de tránsito que incrementan el riesgo de resbalón.",
        "consecuencias": [
            "Caída al mismo nivel con esguinces, contusiones o fracturas",
            "Golpe contra equipos/estanterías",
            "Contaminación cruzada o daño de materiales"
        ],
        "controles": {
            "Eliminación": [
                "Retirar de inmediato el derrame con material absorbente y limpiar la superficie.",
                "Eliminar la fuente del derrame (fugas, recipientes sin tapa)."
            ],
            "Ingeniería": [
                "Cubiertas anti-derrame, bandejas de contención y tapetes antideslizantes en zonas críticas.",
                "Mejorar drenaje y nivelación de piso; usar acabados antideslizantes."
            ],
            "Administrativos": [
                "Procedimiento de respuesta a derrames y estaciones de limpieza visibles.",
                "Programa 5S y rondas de inspección programadas.",
                "Capacitación en reporte inmediato de derrames."
            ],
            "Señalización": [
                "Conos y señal de ‘Piso mojado’ hasta restablecer condiciones seguras.",
                "Demarcación de pasillos y áreas húmedas recurrentes."
            ],
            "EPP": [
                "Calzado con suela antideslizante; guantes resistentes a químicos si aplica."
            ],
            "Normas": [
                "Decreto 1072 de 2015 y Resolución 0312 de 2019 (SG-SST: control de riesgos locativos)."
            ],
            "Conclusión": [
                "Eliminar la condición de inmediato y atacar la causa raíz para evitar recurrencias."
            ]
        }
    }
]       
    }
    return mapping.get(image_hash, [])
