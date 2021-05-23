

FIELD_TYPES = {
    "general": 0,
    "optional": 1,
    "amount": 2,
    "date": 3
}

FIELDS = dict()
FIELDS["shipper"] = FIELD_TYPES["general"] 
FIELDS["consignee"] = FIELD_TYPES["general"] 
FIELDS["place_of_issue"] = FIELD_TYPES["general"] 
FIELDS["waybill_number"] = FIELD_TYPES["general"]
FIELDS["invoice_date"] = FIELD_TYPES["date"] 
FIELDS["marks_and_nos_container_and_seals"] = FIELD_TYPES["general"] 
FIELDS["tare"] = FIELD_TYPES["amount"] 
FIELDS["no_and_kind_of_packages"] = FIELD_TYPES["general"]
FIELDS["total_packages"] = FIELD_TYPES["general"] 
FIELDS["measurement"] = FIELD_TYPES["amount"] 
FIELDS["no_and_kind_of_packages"] = FIELD_TYPES["general"]
FIELDS["gross_weight_cargo"] = FIELD_TYPES["amount"] 
FIELDS["description"] = FIELD_TYPES["general"]
