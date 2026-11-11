"""Headless OpenRocket 23.09 driver (JPype): start the JVM, build the TVC sounding rocket, run simulations.

Used by run_openrocket.py (reference sim + parameter export) and trade_study.py (motor candidates).
"""
import math
import os
import time
from pathlib import Path

import jpype
import jpype.imports

HERE = Path(__file__).resolve().parent
JAR = HERE / "bin" / "OpenRocket-23.09.jar"

# ---------------------------------------------------------------- launch site (Midland, TX)
LAUNCH = dict(lat_deg=31.99, lon_deg=-102.08, elevation_m=850.0,
              rail_length_m=5.4864, rail_angle_deg=6.0, rail_azimuth_deg=90.0)
WIND_MPS = 4.0          # steady, no turbulence (matches a steady-wind 6-DOF case)
WIND_FROM_DEG = 90.0    # wind blows FROM the east, i.e. rail is tilted into the wind (standard practice)

# ---------------------------------------------------------------- airframe (all SI)
D = 0.1567              # 6" fiberglass airframe: OD 6.17 in
R = D / 2
WALL = 0.00216          # 0.085 in wall
DESIGN = dict(
    nose_len=0.65, nose_shoulder=0.15, nose_thick=0.0025,   # Von Karman (LD-Haack), 4.1 cal
    payload_len=0.35, main_len=0.45, avbay_len=0.20, drogue_len=0.30, booster_extra=0.20,  # booster = motor + extra
    fin_count=4, fin_root=0.30, fin_tip=0.10, fin_span=0.155, fin_sweep=0.20, fin_thick=0.00635,
    payload_kg=4.0, avionics_kg=1.4, tvc_kg=1.2,
    main_chute_d=2.44, main_chute_kg=0.85, drogue_d=0.91, drogue_kg=0.25, main_harness_kg=0.40, drogue_harness_kg=0.30,
    ballast_kg=0.0,
)

_started = False


def start():
    """Start the JVM and the minimal OpenRocket Guice application (same modules as SwingStartup)."""
    global _started
    if _started:
        return
    if not JAR.exists():
        raise FileNotFoundError(f"{JAR} missing - run openrocket/setup.sh")
    jpype.startJVM("-Djava.awt.headless=true", f"-Dlogback.configurationFile={HERE / 'logback.xml'}",
                   classpath=[str(JAR)])
    from net.sf.openrocket.startup import Application, GuiModule
    from net.sf.openrocket.plugin import PluginModule
    from com.google.inject import Guice
    gm = GuiModule()
    Application.setInjector(Guice.createInjector(gm, PluginModule()))
    gm.startLoader()
    # The blocking DB provider pops a Swing progress dialog (HeadlessException) until loading finishes: poll.
    for _ in range(240):
        try:
            Application.getMotorSetDatabase()
            Application.getComponentPresetDao()
            break
        except Exception:
            time.sleep(0.25)
    else:
        raise RuntimeError("OpenRocket databases did not load")
    _started = True


def find_motor(manufacturer, designation):
    from net.sf.openrocket.startup import Application
    for s in Application.getMotorSetDatabase().getMotorSets():
        if str(s.getManufacturer().getSimpleName()) == manufacturer and str(s.getDesignation()) == designation:
            # several digitisations may exist; take the first (thrustcurve.org's preferred cert data)
            return s.getMotors()[0]
    raise KeyError(f"{manufacturer} {designation}")


def m_motors(manufacturers=("AeroTech", "Cesaroni Technology")):
    from net.sf.openrocket.startup import Application
    out = []
    for s in Application.getMotorSetDatabase().getMotorSets():
        if str(s.getManufacturer().getSimpleName()) in manufacturers and str(s.getCommonName()).startswith("M"):
            out.append(s.getMotors()[0])
    return out


def _material(kind, name):
    from net.sf.openrocket.database import Databases
    from net.sf.openrocket.material import Material
    return Databases.findMaterial(getattr(Material.Type, kind), name)


def _mass(parent, name, kg, length, offset, ctype="MASSCOMPONENT"):
    from net.sf.openrocket.rocketcomponent import MassComponent
    from net.sf.openrocket.rocketcomponent.position import AxialMethod
    m = MassComponent(length, R - WALL - 0.002, kg)
    m.setName(name)
    m.setMassComponentType(getattr(MassComponent.MassComponentType, ctype))
    m.setAxialMethod(AxialMethod.TOP)
    m.setAxialOffset(offset)
    parent.addChild(m)
    return m


def _bulkhead(parent, name, offset, thick=0.0095):
    from net.sf.openrocket.rocketcomponent import Bulkhead
    from net.sf.openrocket.rocketcomponent.position import AxialMethod
    b = Bulkhead()
    b.setName(name)
    b.setLength(thick)
    b.setOuterRadiusAutomatic(True)
    b.setMaterial(_material("BULK", "Plywood (birch)"))
    b.setAxialMethod(AxialMethod.TOP)
    b.setAxialOffset(offset)
    parent.addChild(b)


def _tube(stage, name, length, glass):
    from net.sf.openrocket.rocketcomponent import BodyTube
    t = BodyTube(length, R, WALL)
    t.setName(name)
    t.setMaterial(glass)
    stage.addChild(t)
    return t


def build(motor, d=None):
    """Return (document, rocket, motor_mount, fcid) for the given OpenRocket Motor."""
    from net.sf.openrocket.document import OpenRocketDocumentFactory
    from net.sf.openrocket.rocketcomponent import (Rocket, AxialStage, NoseCone, Transition, InnerTube,
                                                   TrapezoidFinSet, FinSet, CenteringRing, Parachute, ShockCord,
                                                   RailButton, DeploymentConfiguration, FlightConfigurationId)
    from net.sf.openrocket.rocketcomponent.position import AxialMethod
    from net.sf.openrocket.motor import MotorConfiguration, Motor
    d = {**DESIGN, **(d or {})}
    glass = _material("BULK", "Fiberglass")
    cord = _material("LINE", "Tubular nylon (11 mm, 7/16 in)")

    rocket = Rocket()
    rocket.setName("TVC Sounding Rocket")
    rocket.setDesigner("OE5310 GNC project")
    stage = AxialStage()
    stage.setName("Sustainer")
    rocket.addChild(stage)

    nose = NoseCone(Transition.Shape.HAACK, d["nose_len"], R)
    nose.setShapeParameter(0.0)  # 0 = Von Karman (LD-Haack)
    nose.setName("Nose cone (Von Karman)")
    nose.setThickness(d["nose_thick"])
    nose.setMaterial(glass)
    nose.setAftShoulderLength(d["nose_shoulder"])
    nose.setAftShoulderRadius(R - WALL)
    nose.setAftShoulderThickness(d["nose_thick"])
    nose.setAftShoulderCapped(True)
    stage.addChild(nose)
    if d["ballast_kg"] > 0:
        _mass(nose, "Nose trim ballast", d["ballast_kg"], 0.05, d["nose_len"] - 0.12)

    pay = _tube(stage, "Payload bay", d["payload_len"], glass)
    _mass(pay, "Payload (4 kg)", d["payload_kg"], 0.20, 0.03, "PAYLOAD")
    _bulkhead(pay, "Payload bulkhead", d["payload_len"] - 0.0095)

    main = _tube(stage, "Main parachute bay", d["main_len"], glass)
    chute = Parachute()
    chute.setName("Main parachute")
    chute.setDiameter(d["main_chute_d"])
    chute.setCD(2.2)  # toroidal (e.g. Fruity Chutes IFC), OpenRocket CD is based on canopy diameter
    chute.setLength(0.25)
    chute.setOverrideMass(d["main_chute_kg"])
    chute.setMassOverridden(True)
    chute.setAxialMethod(AxialMethod.TOP)
    chute.setAxialOffset(0.03)
    main.addChild(chute)
    dc = chute.getDeploymentConfigurations().getDefault()
    dc.setDeployEvent(DeploymentConfiguration.DeployEvent.ALTITUDE)
    dc.setDeployAltitude(300.0)
    sc = ShockCord()
    sc.setName("Main harness")
    sc.setMaterial(cord)
    sc.setCordLength(9.0)
    sc.setLength(0.12)
    sc.setOverrideMass(d["main_harness_kg"])
    sc.setMassOverridden(True)
    sc.setAxialMethod(AxialMethod.TOP)
    sc.setAxialOffset(0.30)
    main.addChild(sc)

    av = _tube(stage, "Avionics bay (switch band)", d["avbay_len"], glass)
    _bulkhead(av, "AV fwd bulkhead", 0.0)
    _mass(av, "Avionics (FC, IMU, GPS, batteries, sled)", d["avionics_kg"], 0.20, 0.025, "FLIGHTCOMPUTER")
    _bulkhead(av, "AV aft bulkhead", d["avbay_len"] - 0.0095)

    drg = _tube(stage, "Drogue bay", d["drogue_len"], glass)
    dchute = Parachute()
    dchute.setName("Drogue parachute")
    dchute.setDiameter(d["drogue_d"])
    dchute.setCD(1.5)
    dchute.setLength(0.10)
    dchute.setOverrideMass(d["drogue_kg"])
    dchute.setMassOverridden(True)
    dchute.setAxialMethod(AxialMethod.TOP)
    dchute.setAxialOffset(0.03)
    drg.addChild(dchute)
    dchute.getDeploymentConfigurations().getDefault().setDeployEvent(DeploymentConfiguration.DeployEvent.APOGEE)
    sc2 = ShockCord()
    sc2.setName("Drogue harness")
    sc2.setMaterial(cord)
    sc2.setCordLength(9.0)
    sc2.setLength(0.10)
    sc2.setOverrideMass(d["drogue_harness_kg"])
    sc2.setMassOverridden(True)
    sc2.setAxialMethod(AxialMethod.TOP)
    sc2.setAxialOffset(0.16)
    drg.addChild(sc2)

    mlen = float(motor.getLength())
    d["booster_len"] = mlen + d["booster_extra"]
    boost = _tube(stage, "Booster / fin can", d["booster_len"], glass)
    mrad = float(motor.getDiameter()) / 2
    mount = InnerTube()
    mount.setName("Motor mount / TVC gimbal sleeve")
    mount.setOuterRadius(mrad + 0.0025)
    mount.setThickness(0.0015)
    mount.setLength(mlen + 0.02)
    mount.setMaterial(glass)
    mount.setAxialMethod(AxialMethod.BOTTOM)
    mount.setAxialOffset(0.0)
    mount.setMotorMount(True)
    mount.setMotorOverhang(0.0)
    boost.addChild(mount)
    for i, off in enumerate((0.0, 0.5, 1.0)):
        cr = CenteringRing()
        cr.setName(f"Centering ring {i + 1}")
        cr.setLength(0.006)
        cr.setMaterial(_material("BULK", "Plywood (birch)"))
        cr.setInnerRadiusAutomatic(True)
        cr.setOuterRadiusAutomatic(True)
        cr.setAxialMethod(AxialMethod.BOTTOM)
        cr.setAxialOffset(-off * (mlen - 0.006))
        boost.addChild(cr)
    # TVC gimbal ring, 2 servos, linkages and servo battery, around the aft end of the motor (pivot at the throat)
    _mass(boost, "TVC gimbal mount + servos", d["tvc_kg"], 0.12, d["booster_len"] - 0.14, "MASSCOMPONENT")
    _bulkhead(boost, "Thrust plate / motor retention", d["booster_len"] - mlen - 0.035, 0.0127)

    fins = TrapezoidFinSet(d["fin_count"], d["fin_root"], d["fin_tip"], d["fin_sweep"], d["fin_span"])
    fins.setName("Fins (clipped delta)")
    fins.setThickness(d["fin_thick"])
    fins.setCrossSection(FinSet.CrossSection.AIRFOIL)
    fins.setMaterial(_material("BULK", "Fiberglass"))
    fins.setAxialMethod(AxialMethod.BOTTOM)
    fins.setAxialOffset(-0.01)
    fins.setFilletRadius(0.008)
    fins.setFilletMaterial(_material("BULK", "Fiberglass"))
    boost.addChild(fins)

    for name, off in (("Rail button fwd", 0.20), ("Rail button aft", d["booster_len"] - 0.06)):
        rb = RailButton(0.0101, 0.0101)  # 1515 buttons
        rb.setName(name)
        rb.setAxialMethod(AxialMethod.TOP)
        rb.setAxialOffset(off)
        rb.setAngleOffset(math.pi / 4)  # between fins
        boost.addChild(rb)

    doc = OpenRocketDocumentFactory.createDocumentFromRocket(rocket)
    fcid = FlightConfigurationId()
    rocket.createFlightConfiguration(fcid)
    mc = MotorConfiguration(mount, fcid)
    mc.setMotor(motor)
    mc.setEjectionDelay(Motor.PLUGGED_DELAY)  # TVC flight computer fires the charges
    mount.setMotorConfig(mc, fcid)
    rocket.setSelectedConfiguration(fcid)
    rocket.getFlightConfiguration(fcid).setName(str(motor.getDesignation()))
    rocket.enableEvents()
    rocket.update()
    return doc, rocket, mount, fcid


def make_sim(doc, rocket, fcid, wind=WIND_MPS, name="Midland TX - reference"):
    from net.sf.openrocket.document import Simulation
    s = Simulation(doc, rocket)
    s.setName(name)
    s.setFlightConfigurationId(fcid)
    o = s.getOptions()
    o.setLaunchRodLength(LAUNCH["rail_length_m"])
    o.setLaunchIntoWind(False)
    o.setLaunchRodAngle(math.radians(LAUNCH["rail_angle_deg"]))
    o.setLaunchRodDirection(math.radians(LAUNCH["rail_azimuth_deg"]))
    o.setLaunchLatitude(LAUNCH["lat_deg"])
    o.setLaunchLongitude(LAUNCH["lon_deg"])
    o.setLaunchAltitude(LAUNCH["elevation_m"])
    o.setISAAtmosphere(True)
    o.setWindSpeedAverage(wind)
    o.setWindTurbulenceIntensity(0.0)
    o.setWindDirection(math.radians(WIND_FROM_DEG))
    o.setTimeStep(0.01)
    o.setRandomSeed(42)
    return s


def run(sim):
    """Simulate and return (branch, summary dict)."""
    from net.sf.openrocket.simulation import FlightDataType as T, FlightEvent
    sim.simulate()
    b = sim.getSimulatedData().getBranch(0)

    def ev(name):
        e = b.getFirstEvent(getattr(FlightEvent.Type, name))
        return None if e is None else float(e.getTime())

    t = [float(x) for x in b.get(T.TYPE_TIME)]
    t_rail = ev("LAUNCHROD")
    i_rail = min(range(len(t)), key=lambda i: abs(t[i] - t_rail))
    v = b.get(T.TYPE_VELOCITY_TOTAL)
    stab = [float(x) if x is not None and not math.isnan(float(x)) else float("nan") for x in b.get(T.TYPE_STABILITY)]
    t_bo = ev("BURNOUT")
    powered_coast = [s for ti, s in zip(t, stab) if t_rail <= ti <= (ev("APOGEE") - 3.0) and not math.isnan(s)]
    summary = dict(
        apogee_m_agl=float(b.getMaximum(T.TYPE_ALTITUDE)),
        max_velocity_mps=float(b.getMaximum(T.TYPE_VELOCITY_TOTAL)),
        max_mach=float(b.getMaximum(T.TYPE_MACH_NUMBER)),
        max_accel_mps2=float(b.getMaximum(T.TYPE_ACCELERATION_TOTAL)),
        off_rail_velocity_mps=float(v[i_rail]),
        stability_cal_off_rail=stab[i_rail],
        stability_min_cal=min(powered_coast), stability_max_cal=max(powered_coast),
        time_to_apogee_s=ev("APOGEE"), t_rail_exit=t_rail, t_burnout=t_bo,
        events={str(e.getType().name()): float(e.getTime()) for e in b.getEvents()},
    )
    return b, summary


def load_and_run(path):
    """Load an .ork file (as the GUI would) and re-run its first simulation; returns the run() summary."""
    start()
    from java.io import File
    from net.sf.openrocket.file import GeneralRocketLoader
    doc = GeneralRocketLoader(File(str(path))).load()
    return run(doc.getSimulation(0))[1]
