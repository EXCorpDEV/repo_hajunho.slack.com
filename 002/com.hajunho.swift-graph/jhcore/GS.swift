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
        if(!GS.shared.logLevel.contains(.none)) {
            print("pwd_", String(describing: x.self))
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
    static let none = _logLevel(rawValue: 1 << 8)
    static let network = _logLevel(rawValue: 1 << 9)
    static let all: _logLevel = [.critical, .major, .minor, .just, .graph, .graph2, .graphPanel, .dashboard]
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
    var logLevel : _logLevel
    
    private init() {
//        logLevel = .all
//        logLevel = .network
        logLevel = .none
        //        logLevel = .critical
        //        logLevel = .graphPanel
        //        logLevel = .dashboard
        //        logLevel = .just
    }
    
    static let shared = GS()
}
