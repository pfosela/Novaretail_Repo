import pytest

from novaretail.external_api import ExternalSourceError, parse_response
from novaretail.ingest import parse_eur, parse_invoice_text


def test_parse_eur_formato_espanol():
    assert parse_eur("1.234,56 €") == 1234.56
    assert parse_eur("20,99 €") == 20.99


def test_parse_invoice_text():
    text = ("Deporte Total Iberia S.L.\nCIF: B45678907\nFACTURA Nº F2025-1073\nFecha: 04/10/2025\n"
            "Base imponible: 11.374,09 €\nIVA (21%): 2.388,56 €\nTOTAL: 13.762,65 €")
    r = parse_invoice_text(text)
    assert r["invoice_number"] == "F2025-1073" and r["supplier_cif"] == "B45678907"
    assert r["base_eur"] == 11374.09 and r["vat_rate"] == 0.21 and r["total_eur"] == 13762.65


def test_contrato_api_rechaza_valores_imposibles():
    payload = {"daily": {"time": ["2025-01-01", "2025-01-02"], "temperature_2m_max": [12.0, 80.0],
                         "temperature_2m_min": [3.0, 5.0], "precipitation_sum": [0.0, 1.2]}}
    ok, rejected = parse_response("Madrid", payload)
    assert len(ok) == 1 and rejected == 1


def test_contrato_api_detecta_cambio_de_formato():
    with pytest.raises(ExternalSourceError):
        parse_response("Madrid", {"hourly": {}})
