from novaretail.privacy import mask_text


def test_enmascara_dni_email_telefono():
    t = mask_text("Soy Ana, DNI 12345678Z, ana.ruiz@gmail.com, tel 612345678")
    assert "12345678Z" not in t and "@gmail.com" not in t and "612345678" not in t
