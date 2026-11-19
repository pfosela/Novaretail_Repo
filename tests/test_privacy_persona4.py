"""Pruebas de privacidad y triaje (Unit 17, Persona 4)."""
import pandas as pd
import pytest

from novaretail.privacy import (
    audit_masking, classify_ticket, classify_tickets,
    evaluate_classification, link_ticket_segments, mask_text,
    mask_tickets, personal_data_inventory,
)


def test_identificadores_personales_y_tarjeta():
    text = ("Hola, soy Ana Pérez López. DNI 12345678Z, ana.prueba@example.com, "
            "tel. 612345678, IBAN ES9121000418450200051332 y tarjeta terminada en 1234")
    masked = mask_text(text)
    for value in ("Ana Pérez López", "12345678Z", "@example.com", "612345678", "ES912", "1234"):
        assert value not in masked
    for label in ("[NOMBRE]", "[DNI]", "[EMAIL]", "[TELEFONO]", "[IBAN]", "[TARJETA]"):
        assert label in masked


@pytest.mark.parametrize('address', [
    "Necesito cambiar la dirección de entrega del pedido NR0012345 a C. de Primitiva Ruiz 8, 46998 Oviedo. Gracias",
    "La nueva es Vial de Mariana Conesa 93, 15687 Granada. Teléfono 676851588",
    "La nueva es Cañada de Fernanda Huerta 864 Puerta 5 , 26223 Málaga",
])
def test_enmascara_direcciones_varias(address):
    masked = mask_text(address)
    assert "[DIRECCION]" in masked
    assert not any(cp in masked for cp in ('46998', '15687', '26223'))
    assert "NR0012345" in masked if 'NR0012345' in address else True


def test_no_enmascara_referencia_de_pedido_como_iban():
    assert "NR0000012" in mask_text("Pedido NR0000012; solicito información")


def test_clientes_vinculados_seudonimizados():
    clients = pd.DataFrame([{"customer_id": "C000001", "first_name": "Lucía", "last_name": "García Rubio",
                             "email": "lucia@example.com", "phone": "612345678", "segment": "premium"}])
    orders = pd.DataFrame([{"order_id": "NR0000001", "customer_id": "C000001"}])
    tickets = pd.DataFrame([{"ticket_id": "T1", "channel": "chat", "text":
                             "El pedido NR0000001 se retrasa. Lucía García Rubio, 612345678"}])
    cleaned = mask_tickets(tickets, clients, orders)
    assert "text" not in cleaned.columns
    assert "Lucía García Rubio" not in cleaned.iloc[0]["text_masked"]
    assert "[NOMBRE]" in cleaned.iloc[0]["text_masked"]
    with_segment = link_ticket_segments(cleaned, orders, clients)
    assert with_segment.iloc[0]["segmento"] == "premium"
    audited = audit_masking(tickets, cleaned, clients, orders)
    assert int(audited["sin_enmascarar"].sum()) == 0


@pytest.mark.parametrize('message,cat,priority', [
    ('Solicito la supresión de mis datos conforme al RGPD. [EMAIL]', 'proteccion_datos', 'alta'),
    ('Me habéis cobrado dos veces el pedido NR0000001. [IBAN]', 'cobro_duplicado', 'alta'),
    ('Quiero cambiar la dirección de entrega del pedido NR0000001 a [DIRECCION]', 'cambio_direccion', 'alta'),
    ('Llevo 8 días esperando el pedido NR0000001, seguimiento del transportista', 'retraso_entrega', 'alta'),
    ('Hola, devolví el pedido NR0000001 hace 3 días y aún no tengo reembolso', 'devolucion', 'baja'),
    ('¿Cuándo volveréis a tener stock?', 'consulta_producto', 'baja'),
    ('No funciona desde el primer día. Pido una solución', 'producto_defectuoso', 'media'),
    ('Texto ininteligible', 'sin_clasificar', 'alta'),
])
def test_prioridad_y_categoria(message, cat, priority):
    assert classify_ticket(message) == (cat, priority)


def test_solo_clasifica_ticket_previamente_enmascarado():
    with pytest.raises(ValueError):
        classify_tickets(pd.DataFrame([{"ticket_id": "T1", "text": "DNI 12345678Z"}]))


def test_evaluacion_calcula_recall_y_errores():
    pred = pd.DataFrame([{"ticket_id": "T1", "channel": "chat", "segmento": "estandar", "pred_category": "proteccion_datos", "pred_priority": "alta"},
                         {"ticket_id": "T2", "channel": "email", "segmento": "premium", "pred_category": "devolucion", "pred_priority": "baja"}])
    labels = pd.DataFrame([{"ticket_id": "T1", "category": "proteccion_datos", "priority": "alta"},
                           {"ticket_id": "T2", "category": "devolucion", "priority": "media"}])
    report = evaluate_classification(pred, labels)
    assert report["acierto_categoria_pct"] == 100
    assert report["acierto_prioridad_pct"] == 50
    assert report["recall_alta_pct"] == 100


def test_inventario_campos_sin_revelar_registros():
    inventory = personal_data_inventory()
    assert {"campo", "conjunto", "categoria", "finalidad", "base_legal", "conservacion", "acceso"}.issubset(inventory.columns)
    assert {"customers", "tickets", "invoices"}.issubset(set(inventory["conjunto"]))
