//
//  GlobalSettings.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 15..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

//Where am I, exactly  ex) "".pwd(self)
extension String {
    func pwd(_ x: Any)  {
        if(!GS.s.logLevel.contains(.none)) {
            if x is String {
                print("pwd_\(x)")
            } else {
                print("pwd_", String(describing: x.self))
            }
        }
    }
    
    func pwdJustString(_ x: Any) -> String {
        return String(describing: x.self)
    }
    
    func font1() -> String {
        return "NanumSquareOTFR"
    }
    
    func font1L() -> String {
        return "NanumSquareOTFL"
    }
    
    func font1B() -> String {
        return "NanumSquareOTFB"
    }
    
    func font2() -> String {
        return "Pecita"
    }
}

struct _logLevel: OptionSet {
    let rawValue: Int
    static let critical    = _logLevel(rawValue: 1 << 0)
    static let major  = _logLevel(rawValue: 1 << 1)
    static let minor   = _logLevel(rawValue: 1 << 2)
    static let just   = _logLevel(rawValue: 1 << 3)
    static let graph = _logLevel(rawValue: 1 << 4)
    static let graph2 = _logLevel(rawValue: 1 << 5)
    static let graphPanel = _logLevel(rawValue: 1 << 6)
    static let dashboard = _logLevel(rawValue: 1 << 7)
    static let network = _logLevel(rawValue: 1 << 8)
    static let network2 = _logLevel(rawValue: 1 << 9)
    static let none = _logLevel(rawValue: 1 << 10)
    static let layer = _logLevel(rawValue: 1 << 11)
    static let json = _logLevel(rawValue: 1 << 12)
    static let all: _logLevel = [.critical, .major, .minor, .just, .graph, .graph2, .graphPanel, .dashboard, .network, network2, layer, json]
}

struct _plistV1 : Codable {
    var col1 : Date
    var col2 : Float
}

struct plistV1 : Codable {
    var raw1 : [_plistV1]?
    init(from decoder : Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        raw1 = try values.decodeIfPresent([_plistV1].self, forKey: .raw1)
    }
}

class GS {
    
    static let s = GS()
    
    private init() {
        //                logLevel = .all
        logLevel = .json
        //        logLevel = .layer
        //        logLevel = .none
        //        logLevel = .critical
        //        logLevel = .graphPanel
        //        logLevel = .dashboard
        //        logLevel = .just
    }
    
    /// Margins
    let jhAMarginCommonV : CGFloat = 300 //Axes Virtual Margin(10000,10000)
    
    var jhLMarginCommon : CGFloat = 0 //layer's margin left, right, top, bottom
    var jhLMarginLeft : CGFloat = 0
    var jhLMarginRight : CGFloat = 0
    var jhLMarginTop : CGFloat = 0
    var jhLMarginBottom : CGFloat = 0
    var jhLMarginX : CGFloat {
        get {
            return jhLMarginCommon + jhLMarginLeft
        }
    }
    var jhLMarginY : CGFloat {
        get {
            return jhLMarginCommon + jhLMarginBottom
        }
    }
    
    var jhPSpacing : CGFloat = 30//spacing between each panels
    
    /// Axis
    let jhATextPanelSize : CGFloat = 14 //point
    let jhATextSize : CGFloat = 12 //point
    var jhLBackgroundColor : UIColor = #colorLiteral(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
    
    var logLevel : _logLevel
    var currentServerTime : Double = 0
    var sceneWidthByTime : CGFloat = 86400
    
    var testDataMaxValue : CGFloat = 0
    var testDataMinValue : CGFloat = 0
    var testDataVerticalRatioToDraw_view : CGFloat = 0
}
