//
//  GlobalSettings.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 15..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

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

class GlobalSettings {
    var logLevel : _logLevel
    
    private init() {
        logLevel = .all
        logLevel = .none
        //        logLevel = .critical
        //        logLevel = .graphPanel
        //        logLevel = .dashboard
        //        logLevel = .just
    }
    
    enum eoGraphType {
        case general
        case first
    }
    
    let current_eoGraphType : eoGraphType = eoGraphType.first
    
    static let shared = GlobalSettings()
}
