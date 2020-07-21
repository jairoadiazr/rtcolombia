# Rt Colombia COVID-19

En este reporte presentamos una estimación del número básico de reproducción temporal *Rt* de la epidemia del virus SARS-CoV-2 en Colombia, sus departamentos y municipios.

El número básico de reproducción temporal de una epidemia viral *Rt* lo definimos como el número promedio de personas que contagia una persona infectada en el lapso de tiempo que permaneció infecciosa.

Si *Rt* es mayor que 1 la epidemia crece en número de infectados. Si *Rt* es menor que 1 la epidemia decrece.

[**Datos en tiempo real**](
https://www.datos.gov.co/Salud-y-Protecci-n-Social/Casos-positivos-de-COVID-19-en-Colombia/gt2j-8ykr/data) 

## Tablero de visualización

Ver [aquí](https://dashboard.rtcolombia.com/) la última implementación del tablero.

## Prerequisitos

- Git
- Python3

## Instalación

```bash
git clone https://github.com/jairoadiazr/rtcolombia.git
pip install -r requirements.txt
```

## Visualización local

Para poder visualizar el aplicativo, es necesario editar localmente la última línea del archivo `app.py`.

Reemplazar

```python
if __name__ == '__main__':
    app.run_server(debug=False)
```

por

```python
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
```

Luego ejecutar en el directorio de instalación

```bash
python app.py
```

Finalmente, visualizar el tablero en http://localhost:8050/

## Documentación

El documento que contextualiza y explica el modelo se puede consultar en:
http://rtcolombia.com/

## Autores

- Jairo A. Diaz, División de Ciencias Básicas, Universidad del Norte, Barranquilla. jairoadiazr@gmail.com
- Jairo J. Espinosa, Facultad de Minas, Universidad Nacional de Colombia, Medellín. jairo.espinosa@gmail.com
- Héctor López, Barranquilla. hectorlopezl@outlook.com
- Bernardo Uribe, División de Ciencias Básicas, Universidad del Norte, Barranquilla. buribe@gmail.com
