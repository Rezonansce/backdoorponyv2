from backdoorpony.extensions import db

default_table = db.Table('default_table', db.Model.metadata, db.Column('action_id', db.Integer, db.ForeignKey(
    'action.id')), db.Column('param_id', db.Integer, db.ForeignKey('param.id')))

config_table = db.Table('config_table', db.Model.metadata, db.Column('action_id', db.Integer, db.ForeignKey(
    'action.id')), db.Column('input_id', db.Integer, db.ForeignKey('input.id')))


class Action(db.Model):
    # __tablename__ = 'action'
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    name = db.Column(db.String(20), nullable=False)
    info = db.Column(db.String(500), nullable=False)
    link = db.Column(db.String(200))
    is_attack = db.Column(db.Boolean, nullable=False)
    defaults = db.relationship('Param', secondary=default_table, backref=db.backref('defaults'))
    configs = db.relationship('Input', secondary=config_table, backref=db.backref('configs'))

    def __repr__(self):
        return f"Action('{self.id}', '{self.name}', '{self.info}')"

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class Param(db.Model):
    # __tablename__ = 'param'
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    name = db.Column(db.String(20), nullable=False)
    info = db.Column(db.String(500), nullable=False)
    value = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return f"Param('{self.id}', '{self.name}', '{self.value}')"

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class Input(db.Model):
    #__tablename__ = 'input'
    id = db.Column(db.Integer, primary_key=True, nullable=False)
    input_type = db.Column(db.String(20), nullable=False)

    def __repr__(self):
        return f"Input('{self.id}', '{self.input_type}')"

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
